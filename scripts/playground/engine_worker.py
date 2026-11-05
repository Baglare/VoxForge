# -*- coding: utf-8 -*-
"""Voice Playground engine worker: loads ONE engine once, then serves synthesis requests.

Runs inside that engine's own environment (XTTS/Mock: main .venv, Chatterbox V3 and
VoxCPM2: the isolated VF-Q1 bench venvs). The Playground UI starts at most one worker
at a time, so only one model is ever on the GPU.

Protocol: JSON lines. Requests on stdin, replies on the ORIGINAL stdout prefixed with
``@@VF ``; everything libraries print (Python or C level) is redirected to stderr.

    -> {"cmd": "synth", "id": "...", "chunks": ["..."], "reference": "x.wav",
        "settings": {"min_p": 0.02}, "seed": 1234, "out": "y.wav"}
    <- {"event": "result", "id": "...", "ok": true, "seconds": ..., ...}
    -> {"cmd": "shutdown"}

Engine defaults are the ones used by the VF-Q1 benchmark (scripts/benchmark/run_engine.py);
a request may only add settings the installed library accepts (never dropped silently).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "scripts" / "benchmark"
for _path in (str(REPO_ROOT), str(BENCH_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_engine import Torch, package_versions, save_pcm16, supported_kwargs  # noqa: E402
from sources import chatterbox_pin_problem, load_pins, source_revision  # noqa: E402

PROTOCOL_PREFIX = "@@VF "
ENGINES = ("mock", "xtts", "chatterbox-v3", "voxcpm2")
CHUNK_PAUSE_SECONDS = 0.2
Synth = Callable[[str, Path, Path | None, dict[str, Any], int], dict[str, Any]]


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


def load_voxcpm(device: str) -> tuple[Synth, dict[str, Any]]:
    from voxcpm import VoxCPM

    load_kwargs, _ = supported_kwargs(VoxCPM.from_pretrained, {"load_denoiser": False})
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", **load_kwargs)
    base = {"cfg_value": 2.0, "inference_timesteps": 10}  # VF-Q1 benchmark ayarı
    sample_rate = model.tts_model.sample_rate
    accepts_seed = "seed" in inspect.signature(model.generate).parameters

    def synth(text: str, out_path: Path, reference: Path | None, settings: dict[str, Any], seed: int) -> dict[str, Any]:
        if reference is None:
            raise ValueError("VoxCPM2 referans ses ister.")
        wanted = {**base, **settings, "reference_wav_path": str(reference)}
        kwargs, dropped = supported_kwargs(model.generate, wanted)
        if dropped:
            raise ValueError(f"Kurulu voxcpm bu argümanları kabul etmiyor: {dropped}")
        if accepts_seed:
            kwargs["seed"] = seed
        save_pcm16(out_path, model.generate(text=text, **kwargs), sample_rate)
        return {"applied": {**base, **settings}}

    return synth, {
        "model": "openbmb/VoxCPM2",
        "mode": "controllable (yalnız referans)",
        "sample_rate": sample_rate,
        "defaults": base,
    }


LOADERS: dict[str, Callable[[str], tuple[Synth, dict[str, Any]]]] = {
    "mock": load_mock,
    "xtts": load_xtts,
    "chatterbox-v3": load_chatterbox,
    "voxcpm2": load_voxcpm,
}


# ---- helpers -----------------------------------------------------------------------------
def concat_pcm16(parts: list[Path], out_path: Path, pause_seconds: float = CHUNK_PAUSE_SECONDS) -> float:
    """Join same-format PCM WAV parts with a short pause; returns the duration in seconds."""
    params = None
    frames: list[bytes] = []
    for index, part in enumerate(parts):
        with wave.open(str(part), "rb") as handle:
            current = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if params is None:
                params = current
            elif current != params:
                raise ValueError(f"Parça biçimleri uyuşmuyor: {params} != {current}")
            if index:
                frames.append(b"\x00" * int(params[2] * pause_seconds) * params[0] * params[1])
            frames.append(handle.readframes(handle.getnframes()))
    if params is None:
        raise ValueError("Birleştirilecek parça yok.")
    temporary = out_path.with_name(out_path.name + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(params[0])
        handle.setsampwidth(params[1])
        handle.setframerate(params[2])
        for block in frames:
            handle.writeframes(block)
    os.replace(temporary, out_path)
    with wave.open(str(out_path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


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


def handle_synth(request: dict[str, Any], synth: Synth, torch_helper: Torch) -> dict[str, Any]:
    out_path = Path(request["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [chunk for chunk in request.get("chunks", []) if str(chunk).strip()]
    if not chunks:
        raise ValueError("Boş metin.")
    reference = Path(request["reference"]) if request.get("reference") else None
    settings = dict(request.get("settings") or {})
    seed = int(request.get("seed", 1234))
    parts: list[Path] = []
    applied: dict[str, Any] = {}
    warnings: list[str] = []
    torch_helper.reset_peak()
    started = time.perf_counter()
    try:
        for index, chunk in enumerate(chunks):
            part = out_path.with_name(f"{out_path.stem}.part{index:03d}.wav")
            torch_helper.seed(seed)
            info = synth(chunk, part, reference, settings, seed)
            applied = info.get("applied", applied)
            warnings.extend(info.get("warnings", []))
            parts.append(part)
        torch_helper.sync()
        seconds = time.perf_counter() - started
        audio_seconds = concat_pcm16(parts, out_path)
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
        "seed": seed,
        "applied_settings": applied,
        "warnings": warnings,
        "vram_peak_mb": torch_helper.peak_mb(),
        "vram": vram_snapshot(torch_helper),
    }


def serve(engine: str, channel: Channel) -> int:
    torch_helper = Torch()
    device = "cpu" if engine == "mock" else torch_helper.device  # Mock GPU kullanmaz
    try:
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
    args = parser.parse_args(argv)
    channel = Channel()
    return serve(args.engine, channel)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
