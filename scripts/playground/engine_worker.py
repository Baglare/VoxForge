# -*- coding: utf-8 -*-
"""Voice Playground engine worker: loads ONE engine once, then serves synthesis requests.

Runs inside that engine's own environment (XTTS/Mock: main .venv, Chatterbox V3 and
VoxCPM2: the isolated VF-Q1 bench venvs). The Playground UI starts at most one worker
at a time, so only one model is ever on the GPU.

Protocol: JSON lines. Requests on stdin, replies on the ORIGINAL stdout prefixed with
``@@VF ``; everything libraries print (Python or C level) is redirected to stderr.

    -> {"cmd": "synth", "id": "...", "chunks": ["..."], "reference": "x.wav",
        "reference_text": null, "settings": {"min_p": 0.02}, "seed": 1234, "out": "y.wav",
        "style": null, "pauses": null, "trim": false}
    <- {"event": "result", "id": "...", "ok": true, "seconds": ..., ...}
    -> {"cmd": "shutdown"}

Engine defaults are the ones used by the VF-Q1 benchmark (scripts/benchmark/run_engine.py);
a request may only add settings the installed library accepts (never dropped silently).

Device (VF-015): ``--device cuda`` (default) REQUIRES a working CUDA torch and the model
must report CUDA after loading; there is no silent CPU fallback. ``--device cpu`` exists
only for explicit CPU/GPU comparisons (the manager also hides the GPU from that process).
The ready event carries the runtime (torch build, GPU, dtype, torch.compile state).

VF-016: ``style`` is a VoxCPM2 natural-language control instruction kept apart from the
spoken text; the worker renders it as the model's documented ``(instruction)text`` prefix
on every chunk. It is rejected together with Ultimate Cloning (prompt continuation), like
the library's own CLI does. ``pauses`` (one value per gap) and ``trim`` control how the
chunks are joined (voxforge/playground/audio_join.py); without them the join is the
original 0.2 s one, byte for byte.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "scripts" / "benchmark"
for _path in (str(REPO_ROOT), str(BENCH_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_engine import Torch, package_versions, save_pcm16, supported_kwargs  # noqa: E402
from sources import chatterbox_pin_problem, load_pins, source_revision  # noqa: E402

from voxforge.playground.audio_join import join_wavs  # noqa: E402

PROTOCOL_PREFIX = "@@VF "
ENGINES = ("mock", "xtts", "chatterbox-v3", "voxcpm2")
DEVICES = ("cuda", "cpu")
CHUNK_PAUSE_SECONDS = 0.2
# synth(text, out_path, reference, settings, seed, **extra) -> {"applied", "warnings", "timings"}
# extra: reference_text (VoxCPM2 Ultimate Cloning), only when the request carries one.
Synth = Callable[..., dict[str, Any]]
VOXCPM_BASE = {"cfg_value": 2.0, "inference_timesteps": 10}  # VF-Q1 benchmark ayarı (varsayılan)
VOXCPM_ALLOWED = ("cfg_value", "inference_timesteps", "retry_badcase", "retry_badcase_max_times")
STYLE_MAX_CHARS = 160


def clean_style(style: str | None) -> str | None:
    """A control instruction must not contain parentheses (it would end the prefix early)."""
    if style is None:
        return None
    cleaned = " ".join(str(style).replace("(", " ").replace(")", " ").split())
    if len(cleaned) > STYLE_MAX_CHARS:
        raise ValueError(f"Stil yönergesi en fazla {STYLE_MAX_CHARS} karakter olabilir.")
    return cleaned or None


class Channel:
    """Keeps the real stdout for protocol replies; the process stdout goes to stderr."""

    def __init__(self) -> None:
        sys.stdout.flush()
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        os.dup2(2, 1)  # C extensions printing to fd 1 can no longer corrupt the protocol
        sys.stdout = sys.stderr

    def send(self, payload: dict[str, Any]) -> None:
        self._out.write(PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
        self._out.flush()


# ---- engines: load(device) -> (synth, info) ---------------------------------------------
def _reject_unknown(function: Callable[..., Any], settings: dict[str, Any]) -> None:
    _, dropped = supported_kwargs(function, settings)
    if dropped:
        raise ValueError(f"Kurulu kütüphane bu ayarları kabul etmiyor: {dropped}")


def load_mock(device: str) -> tuple[Synth, dict[str, Any]]:
    from scripts.mock_tts_engine import MockTTSEngine

    engine = MockTTSEngine()

    def synth(text: str, out_path: Path, reference: Path | None, settings: dict[str, Any], seed: int) -> dict[str, Any]:
        if settings:
            raise ValueError(f"Mock motor ayar kabul etmiyor: {sorted(settings)}")
        engine.tts_to_file(text=text, file_path=str(out_path))
        return {}

    return synth, {"model": "mock-sine-v1", "defaults": {}}


def load_xtts(device: str) -> tuple[Synth, dict[str, Any]]:
    # Ürün adaptörü: XTTS çağrısı Studio/demo ve VF-Q1 ile aynı (preset yok = kütüphane varsayılanı).
    from voxforge.engines import SynthesisRequest, create_engine

    engine = create_engine("xtts")

    def synth(text: str, out_path: Path, reference: Path | None, settings: dict[str, Any], seed: int) -> dict[str, Any]:
        result = engine.synthesize(
            SynthesisRequest(text=text, out_path=out_path, language="tr", reference_path=reference, preset=dict(settings))
        )
        return {"applied": result.params, "warnings": result.warnings}

    return synth, {"model": engine.capabilities.model_id, "defaults": {"preset": "kütüphane varsayılanı"}}


def load_chatterbox(device: str) -> tuple[Synth, dict[str, Any]]:
    pin = load_pins()["chatterbox"]
    installed = source_revision(pin["package"])
    problem = chatterbox_pin_problem(installed, pin)
    if problem:
        raise RuntimeError(problem)

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    load_kwargs, dropped = supported_kwargs(ChatterboxMultilingualTTS.from_pretrained, {"device": device, "t3_model": "v3"})
    if "t3_model" in dropped:
        raise RuntimeError("Kurulu chatterbox-tts t3_model='v3' desteklemiyor (setup_env.ps1 -Engine chatterbox).")
    model = ChatterboxMultilingualTTS.from_pretrained(**load_kwargs)

    def synth(text: str, out_path: Path, reference: Path | None, settings: dict[str, Any], seed: int) -> dict[str, Any]:
        if reference is None:
            raise ValueError("Chatterbox referans ses ister.")
        _reject_unknown(model.generate, settings)
        wav = model.generate(text, language_id="tr", audio_prompt_path=str(reference), **settings)
        save_pcm16(out_path, wav, model.sr)
        return {"applied": dict(settings)}

    defaults = {
        name: p.default
        for name, p in inspect.signature(model.generate).parameters.items()
        if p.default is not inspect.Parameter.empty and isinstance(p.default, (int, float))
    }
    return synth, {
        "model": "ResembleAI/chatterbox t3_model=v3",
        "source": f"{pin['repo']}@{installed['commit']}",
        "sample_rate": model.sr,
        "defaults": defaults,
        "note": "Çıktılar PerTh filigranı içerir (kütüphane varsayılanı).",
    }


def voxcpm_compile_choice(device: str) -> tuple[bool, str]:
    """torch.compile only when explicitly requested, on CUDA, with triton importable.

    VoxCPM's own default is optimize=True: without triton (the Windows default) it prints a
    warning, disables itself and still runs a warm-up generation, so it is passed explicitly.
    """
    if os.environ.get("VOXFORGE_VOXCPM_COMPILE", "").strip() != "1":
        return False, "kapalı (varsayılan; VOXFORGE_VOXCPM_COMPILE=1 ile denenir)"
    if device != "cuda":
        return False, "kapalı (yalnız CUDA'da)"
    if importlib.util.find_spec("triton") is None:
        return False, "kapalı (triton kurulu değil)"
    return True, "açık (torch.compile reduce-overhead)"


PROMPT_CACHE_SIZE = 4


def _file_key(path: Any) -> Any:
    if not path:
        return None
    resolved = Path(str(path))
    stat = resolved.stat()
    return (str(resolved.resolve()), stat.st_size, stat.st_mtime_ns)


def memoized_prompt_builder(
    build: Callable[..., Any], stats: dict[str, float], enabled: bool, torch: Any, device: str
) -> Callable[..., Any]:
    """Wrap VoxCPM2 ``build_prompt_cache``: time it and reuse results for the same inputs.

    Encoding a ~30 s reference with the AudioVAE costs ~1.2 GB extra VRAM and up to a few
    seconds. The encoder is deterministic (noise exists only in the decoder) and the result
    lives on the CPU and is only read, so reusing it gives the same audio. Key: prompt text
    plus path, size and mtime of each audio file (an edited file is encoded again).
    """
    cache: dict[Any, Any] = {}

    def wrapped(prompt_text: Any = None, prompt_wav_path: Any = None, reference_wav_path: Any = None, **kwargs: Any) -> Any:
        key = (prompt_text, _file_key(prompt_wav_path), _file_key(reference_wav_path), tuple(sorted(kwargs.items())))
        if enabled and key in cache:
            stats["prompt_cache_hits"] += 1
            return cache[key]
        started = time.perf_counter()
        try:
            result = build(
                prompt_text=prompt_text, prompt_wav_path=prompt_wav_path, reference_wav_path=reference_wav_path, **kwargs
            )
        finally:
            stats["prompt_seconds"] += time.perf_counter() - started
        if device == "cuda" and torch is not None:
            torch.cuda.empty_cache()  # kodlamanın ~1.2 GB geçici belleği LM döngüsünden önce bırakılır
        if enabled:
            while len(cache) >= PROMPT_CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[key] = result
        return result

    return wrapped


def load_voxcpm(device: str) -> tuple[Synth, dict[str, Any]]:
    from voxcpm import VoxCPM

    compile_on, compile_note = voxcpm_compile_choice(device)
    load_kwargs, dropped = supported_kwargs(
        VoxCPM.from_pretrained, {"load_denoiser": False, "device": device, "optimize": compile_on}
    )
    if "device" in dropped:
        raise RuntimeError("Kurulu voxcpm cihaz seçimini desteklemiyor; sessiz CPU'ya geçiş riski nedeniyle durduruldu.")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", **load_kwargs)
    tts = model.tts_model
    actual = str(getattr(tts, "device", device))
    if actual.split(":")[0] != device:
        raise RuntimeError(f"VoxCPM2 {device} istendi ama {actual} üzerinde yüklendi.")
    sample_rate = tts.sample_rate
    # generate(*args, **kwargs) her şeyi kabul eder; gerçek imza _generate'te.
    signature_target = getattr(model, "_generate", model.generate)
    stats = {"prompt_seconds": 0.0, "prompt_cache_hits": 0.0}
    build = getattr(tts, "build_prompt_cache", None)
    use_prompt_cache = os.environ.get("VOXFORGE_VOXCPM_PROMPT_CACHE", "1").strip() != "0"
    torch = Torch().torch
    if build is not None:
        tts.build_prompt_cache = memoized_prompt_builder(build, stats, use_prompt_cache, torch, device)

    def synth(
        text: str,
        out_path: Path,
        reference: Path | None,
        settings: dict[str, Any],
        seed: int,
        reference_text: str | None = None,
        style: str | None = None,
    ) -> dict[str, Any]:
        if reference is None:
            raise ValueError("VoxCPM2 referans ses ister.")
        style = clean_style(style)
        if style and reference_text is not None:
            raise ValueError(
                "Stil yönergesi Ultimate Cloning ile birlikte kullanılamaz (model prompt devamı yapar; "
                "voxcpm CLI de --control ile --prompt-text'i reddeder)."
            )
        unknown = sorted(set(settings) - set(VOXCPM_ALLOWED))
        if unknown:
            raise ValueError(f"VoxCPM2 Playground bu ayarları desteklemiyor: {unknown}")
        wanted = {**VOXCPM_BASE, **settings, "reference_wav_path": str(reference)}
        mode = "controllable (yalnız referans)"
        if reference_text is not None:
            if not reference_text.strip():
                raise ValueError("Ultimate Cloning için referansın birebir transkripti gerekli.")
            # Model kartı: en yüksek benzerlik için aynı klip hem referans hem prompt olarak verilir.
            wanted.update(prompt_wav_path=str(reference), prompt_text=reference_text.strip())
            mode = "ultimate (referans + transkript)"
        kwargs, dropped = supported_kwargs(signature_target, wanted)
        if dropped:
            raise ValueError(f"Kurulu voxcpm bu argümanları kabul etmiyor: {dropped}")
        before = dict(stats)
        model_text = f"({style}){text}" if style else text
        try:
            save_pcm16(out_path, model.generate(text=model_text, **kwargs), sample_rate)
        finally:
            if device == "cuda" and torch is not None:
                # Geçici blokları sürücüye geri ver: 8 GB'ta masaüstü uygulamalarıyla paylaşılan
                # VRAM dolarsa Windows belleği sistem RAM'ine taşır ve üretim 10x+ yavaşlar.
                torch.cuda.empty_cache()
        applied = {**VOXCPM_BASE, **settings, "mode": mode}
        if style:
            applied["style"] = style
        return {
            "applied": applied,
            "timings": {name: stats[name] - before[name] for name in stats},
        }

    return synth, {
        "model": "openbmb/VoxCPM2",
        "mode": "controllable; ultimate isteğe bağlı",
        "sample_rate": sample_rate,
        "defaults": dict(VOXCPM_BASE),
        "dtype": str(getattr(getattr(tts, "config", None), "dtype", "?")),
        "compile": compile_note,
        "vram_policy": (
            "referans kodlaması önbellekte (aynı sonuç) + üretim sonrası empty_cache"
            if use_prompt_cache
            else "önbelleksiz (VOXFORGE_VOXCPM_PROMPT_CACHE=0) + üretim sonrası empty_cache"
        ),
    }


LOADERS: dict[str, Callable[[str], tuple[Synth, dict[str, Any]]]] = {
    "mock": load_mock,
    "xtts": load_xtts,
    "chatterbox-v3": load_chatterbox,
    "voxcpm2": load_voxcpm,
}


# ---- helpers -----------------------------------------------------------------------------
def concat_pcm16(
    parts: list[Path], out_path: Path, pause_seconds: float | list[float] = CHUNK_PAUSE_SECONDS, trim: bool = False
) -> float:
    """Join same-format PCM WAV parts; returns the duration in seconds (see audio_join)."""
    return float(join_wavs(parts, out_path, pauses=pause_seconds, trim=trim)["duration"])


def vram_snapshot(torch_helper: Torch) -> dict[str, Any]:
    torch = torch_helper.torch
    if torch is None or not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        "device_used_mb": round((total - free) / 2**20, 1),
        "device_total_mb": round(total / 2**20, 1),
    }


def runtime_info(torch_helper: Torch) -> dict[str, Any]:
    """What really runs: torch build, CUDA runtime, GPU and its capabilities."""
    torch = torch_helper.torch
    if torch is None:
        return {"torch": None, "cuda_available": False}
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads": torch.get_num_threads(),
    }
    if info["cuda_available"]:
        major, minor = torch.cuda.get_device_capability(0)
        info.update(
            gpu=torch.cuda.get_device_name(0),
            compute_capability=f"{major}.{minor}",
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
        )
    return info


def check_device(engine: str, device: str, torch_helper: Torch) -> None:
    """Refuse to start instead of quietly running a GPU engine on the CPU."""
    if device != "cuda":
        return
    torch = torch_helper.torch
    if torch is None or not torch.cuda.is_available():
        build = torch.__version__ if torch is not None else "yok"
        raise RuntimeError(
            f"CUDA kullanılamıyor (torch {build}); {engine} CPU'ya geçirilmedi. "
            "Onarım: scripts/benchmark/setup_env.ps1 -Engine <ortam> -TorchOnly. "
            "Bilerek CPU karşılaştırması için cihaz olarak CPU seçin."
        )


def handle_synth(request: dict[str, Any], synth: Synth, torch_helper: Torch) -> dict[str, Any]:
    out_path = Path(request["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [chunk for chunk in request.get("chunks", []) if str(chunk).strip()]
    if not chunks:
        raise ValueError("Boş metin.")
    reference = Path(request["reference"]) if request.get("reference") else None
    settings = dict(request.get("settings") or {})
    seed = int(request.get("seed", 1234))
    extra: dict[str, Any] = {}
    if request.get("reference_text") is not None:
        extra["reference_text"] = str(request["reference_text"])
    if request.get("style"):
        extra["style"] = str(request["style"])
    pauses = request.get("pauses")
    if pauses is not None:
        pauses = [float(value) for value in pauses]
        if len(pauses) != len(chunks) - 1:
            raise ValueError(f"{len(chunks)} parça için {len(chunks) - 1} duraklama gerekir.")
    trim = bool(request.get("trim"))
    parts: list[Path] = []
    applied: dict[str, Any] = {}
    warnings: list[str] = []
    chunk_seconds: list[float] = []
    stage_seconds: dict[str, float] = {}
    torch_helper.reset_peak()
    started = time.perf_counter()
    try:
        for index, chunk in enumerate(chunks):
            part = out_path.with_name(f"{out_path.stem}.part{index:03d}.wav")
            torch_helper.seed(seed)
            chunk_started = time.perf_counter()
            info = synth(chunk, part, reference, settings, seed, **extra)
            torch_helper.sync()
            chunk_seconds.append(round(time.perf_counter() - chunk_started, 3))
            applied = info.get("applied", applied)
            warnings.extend(info.get("warnings", []))
            for name, value in (info.get("timings") or {}).items():
                stage_seconds[name] = round(stage_seconds.get(name, 0.0) + float(value), 3)
            parts.append(part)
        torch_helper.sync()
        seconds = time.perf_counter() - started
        audio_seconds = concat_pcm16(parts, out_path, CHUNK_PAUSE_SECONDS if pauses is None else pauses, trim)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    return {
        "ok": True,
        "path": str(out_path),
        "seconds": round(seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "rtf": round(seconds / audio_seconds, 3) if audio_seconds else None,
        "chunks": len(chunks),
        "chunk_seconds": chunk_seconds,
        "stage_seconds": stage_seconds,
        "join": {"pauses": pauses, "trim": trim},
        "seed": seed,
        "applied_settings": applied,
        "warnings": warnings,
        "vram_peak_mb": torch_helper.peak_mb(),
        "vram": vram_snapshot(torch_helper),
    }


def serve(engine: str, channel: Channel, device: str = "cuda") -> int:
    torch_helper = Torch()
    if engine == "mock":
        device = "cpu"  # Mock GPU kullanmaz
    try:
        check_device(engine, device, torch_helper)
        torch_helper.reset_peak()
        started = time.perf_counter()
        synth, info = LOADERS[engine](device)
        torch_helper.sync()
        load_seconds = round(time.perf_counter() - started, 2)
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        channel.send({"event": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
        return 1
    channel.send({
        "event": "ready",
        "engine": engine,
        "device": device,
        "runtime": runtime_info(torch_helper),
        "load_seconds": load_seconds,
        "load_vram_peak_mb": torch_helper.peak_mb(),
        "vram": vram_snapshot(torch_helper),
        "versions": package_versions(),
        **info,
    })
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            channel.send({"event": "result", "id": None, "ok": False, "error": f"Geçersiz istek: {exc}"})
            continue
        if request.get("cmd") == "shutdown":
            break
        if request.get("cmd") != "synth":
            channel.send({"event": "result", "id": request.get("id"), "ok": False, "error": "Bilinmeyen komut"})
            continue
        try:
            reply = handle_synth(request, synth, torch_helper)
        except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM; worker stays alive
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)}
            if torch_helper.torch is not None and torch_helper.torch.cuda.is_available():
                torch_helper.torch.cuda.empty_cache()
        channel.send({"event": "result", "id": request.get("id"), **reply})
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, choices=ENGINES)
    parser.add_argument("--device", default="cuda", choices=DEVICES)
    args = parser.parse_args(argv)
    channel = Channel()
    return serve(args.engine, channel, args.device)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
