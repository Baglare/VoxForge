"""One-engine-at-a-time worker manager for the Voice Playground.

Each engine runs as a long-lived subprocess in its own environment
(``scripts/playground/engine_worker.py``). ``EngineManager`` owns at most one such
process: asking for another engine first shuts the current one down (its CUDA memory
is released with the process), then starts the new one. All calls are serialised by
one lock, so the GPU never holds two models.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "playground" / "engine_worker.py"
PROTOCOL_PREFIX = "@@VF "
LOAD_TIMEOUT_SECONDS = 900.0  # ilk açılışta model önbellekten yüklenir; indirme gerekiyorsa uzun sürebilir
SYNTH_TIMEOUT_SECONDS = 1800.0
SHUTDOWN_GRACE_SECONDS = 20.0


class WorkerError(RuntimeError):
    """Engine worker failed to start, crashed or rejected a request."""


def data_root() -> Path:
    configured = os.environ.get("AVENOX_DATA_ROOT")
    if configured:
        return Path(configured)
    return Path(REPO_ROOT.anchor) / "AvenoxData"


def bench_python(name: str) -> Path:
    """Interpreter of an isolated VF-Q1 environment (created by scripts/benchmark/setup_env.ps1)."""
    return data_root() / "Environments" / f"VoxForge-bench-{name}" / "Scripts" / "python.exe"


def engine_python(engine: str) -> Path:
    if engine in ("xtts", "mock"):
        return Path(sys.executable)  # ana .venv (Studio ile aynı XTTS kurulumu)
    if engine == "chatterbox-v3":
        return bench_python("chatterbox")
    if engine == "voxcpm2":
        return bench_python("voxcpm")
    raise WorkerError(f"Bilinmeyen motor: {engine}")


def _shared_ffmpeg_dir() -> Path | None:
    """Gyan.FFmpeg.Shared bin folder (XTTS TorchCodec needs its DLLs on PATH on Windows)."""
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("ProgramFiles", "")) / "WinGet" / "Packages",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        found = sorted(root.glob("Gyan.FFmpeg.Shared*/**/avcodec-*.dll"), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return found[0].parent
    return None


def prepend_path(path_value: str, directory: Path) -> str:
    """Put ``directory`` FIRST in PATH, removing later copies.

    Being somewhere in PATH is not enough: on this machine TorchCodec only loads when the
    FFmpeg shared bin precedes C:\\Windows\\system32 (run_voice_benchmark.ps1 always prepends).
    """
    target = os.path.normcase(os.path.normpath(str(directory)))
    rest = [
        part for part in path_value.split(os.pathsep)
        if part and os.path.normcase(os.path.normpath(part)) != target
    ]
    return os.pathsep.join([str(directory), *rest])


def worker_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Same model caches and FFmpeg setup as run_voice_benchmark.ps1 / run_gradio_demo.ps1."""
    env = dict(os.environ if base is None else base)
    root = data_root()
    env.setdefault("HF_HOME", str(root / "Models" / "HF"))
    coqui = root / "Models" / "Coqui"
    if "TTS_HOME" not in env and (coqui / "tts").is_dir():
        env["TTS_HOME"] = str(coqui)
    ffmpeg = _shared_ffmpeg_dir()
    if ffmpeg is not None:
        env["PATH"] = prepend_path(env.get("PATH", ""), ffmpeg)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


@dataclass
class WorkerProcess:
    engine: str
    process: subprocess.Popen
    log_path: Path
    replies: "queue.Queue[dict[str, Any] | None]" = field(default_factory=queue.Queue)
    ready: dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.process.poll() is None

    def wait_reply(self, timeout: float, event: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError(f"{self.engine} yanıt vermedi ({timeout:.0f} sn). Günlük: {self.log_path}")
            try:
                reply = self.replies.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if reply is None:
                code = self.process.poll()
                raise WorkerError(f"{self.engine} süreci kapandı (çıkış {code}). Günlük: {self.log_path}")
            if reply.get("event") in (event, "error"):
                return reply

    def send(self, payload: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()


def _pump(worker: WorkerProcess, log_handle: Any) -> None:
    assert worker.process.stdout is not None
    for line in worker.process.stdout:
        if line.startswith(PROTOCOL_PREFIX):
            try:
                worker.replies.put(json.loads(line[len(PROTOCOL_PREFIX):]))
                continue
            except json.JSONDecodeError:
                pass
        log_handle.write(line)
        log_handle.flush()
    worker.process.stdout.close()
    log_handle.close()
    worker.replies.put(None)


class EngineManager:
    """Keeps at most one engine process alive; every call goes through one lock."""

    def __init__(
        self,
        log_dir: Path,
        python_for: Callable[[str], Path] = engine_python,
        env: dict[str, str] | None = None,
        load_timeout: float = LOAD_TIMEOUT_SECONDS,
        synth_timeout: float = SYNTH_TIMEOUT_SECONDS,
        worker_script: Path = WORKER_SCRIPT,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.worker_script = Path(worker_script)
        self.python_for = python_for
        self.env = env
        self.load_timeout = load_timeout
        self.synth_timeout = synth_timeout
        self._lock = threading.Lock()
        self._current: WorkerProcess | None = None
        self.started_count = 0  # testler ve durum satırı için

    # -- state ------------------------------------------------------------------------------
    @property
    def current_engine(self) -> str | None:
        worker = self._current
        return worker.engine if worker is not None and worker.alive() else None

    def status(self) -> dict[str, Any]:
        worker = self._current
        if worker is None or not worker.alive():
            return {"engine": None}
        return {"engine": worker.engine, "pid": worker.process.pid, **{k: worker.ready.get(k) for k in ("device", "model", "load_seconds")}}

    # -- lifecycle --------------------------------------------------------------------------
    def _start(self, engine: str) -> WorkerProcess:
        python = self.python_for(engine)
        if not Path(python).is_file():
            raise WorkerError(
                f"{engine} ortamı bulunamadı: {python}. "
                "Kurulum: scripts/benchmark/setup_env.ps1 (docs/VOICE_QUALITY_BENCHMARK.md)."
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{engine}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {engine} başlatılıyor ({python})\n")
        log_handle.flush()
        process = subprocess.Popen(
            [str(python), "-u", str(self.worker_script), "--engine", engine],
            cwd=str(REPO_ROOT),
            env=worker_env(self.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        worker = WorkerProcess(engine=engine, process=process, log_path=log_path)
        threading.Thread(target=_pump, args=(worker, log_handle), daemon=True).start()
        self.started_count += 1
        try:
            reply = worker.wait_reply(self.load_timeout, "ready")
        except WorkerError:
            self._terminate(worker)
            raise
        if reply.get("event") != "ready":
            self._terminate(worker)
            raise WorkerError(f"{engine} yüklenemedi: {reply.get('error')}. Günlük: {log_path}")
        worker.ready = reply
        return worker

    def _terminate(self, worker: WorkerProcess) -> None:
        if worker.alive():
            try:
                worker.send({"cmd": "shutdown"})
                worker.process.stdin.close()  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass
            try:
                worker.process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        if worker.process.stdin is not None and not worker.process.stdin.closed:
            try:
                worker.process.stdin.close()
            except OSError:
                pass

    def _ensure(self, engine: str) -> tuple[WorkerProcess, bool]:
        worker = self._current
        if worker is not None and worker.engine == engine and worker.alive():
            return worker, False
        if worker is not None:
            self._current = None
            self._terminate(worker)  # önce eski motor kapanır: GPU'da tek model
        self._current = self._start(engine)
        return self._current, True

    def unload(self) -> str | None:
        with self._lock:
            worker, self._current = self._current, None
            if worker is None:
                return None
            self._terminate(worker)
            return worker.engine

    def close(self) -> None:
        self.unload()

    # -- requests ---------------------------------------------------------------------------
    def synthesize(self, engine: str, request: dict[str, Any]) -> dict[str, Any]:
        """Run one request on ``engine``; returns the worker reply plus load info."""
        with self._lock:
            worker, loaded_now = self._ensure(engine)
            request = {"cmd": "synth", "id": f"{time.time_ns()}", **request}
            try:
                worker.send(request)
                reply = worker.wait_reply(self.synth_timeout, "result")
            except (WorkerError, OSError, ValueError) as exc:
                self._current = None
                self._terminate(worker)
                raise WorkerError(str(exc)) from exc
            if reply.get("event") == "error" or not reply.get("ok"):
                raise WorkerError(reply.get("error") or "bilinmeyen hata")
            return {**reply, "loaded_now": loaded_now, "worker": dict(worker.ready)}
