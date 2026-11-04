# -*- coding: utf-8 -*-
"""Generate the benchmark test set with ONE engine, inside that engine's own venv (VF-Q1).

Usage (see docs/VOICE_QUALITY_BENCHMARK.md or run_voice_benchmark.ps1):
    <engine-venv>\\python.exe scripts/benchmark/run_engine.py --run <run_dir> --engine chatterbox-v3

Only the selected engine's libraries are imported. Output goes to <run_dir>/<engine>/;
the reference used is the copy inside the run folder, never the original profile.
Each item is timed separately (model load is reported apart), peak VRAM is recorded,
and one failing item never stops the rest. An existing results.json is not overwritten
unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import os
import platform
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.runs import utc_now_iso, write_json_atomic  # noqa: E402  (stdlib-only helpers)

ENGINE_CHOICES = ("xtts", "chatterbox-v3", "voxcpm2", "voxcpm2-ultimate", "mock")
VERSION_PACKAGES = ("torch", "torchaudio", "chatterbox-tts", "voxcpm", "coqui-tts", "TTS", "transformers", "numpy")


def package_versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for name in VERSION_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def supported_kwargs(function: Callable[..., Any], wanted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only arguments the installed library accepts; report the dropped ones (never silently)."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return dict(wanted), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(wanted), []
    kept = {key: value for key, value in wanted.items() if key in parameters}
    return kept, sorted(set(wanted) - set(kept))


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class Torch:
    """Lazy torch helpers so the Mock path needs nothing but the standard library."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError:
            torch = None
        self.torch = torch

    @property
    def device(self) -> str:
        return "cuda" if self.torch is not None and self.torch.cuda.is_available() else "cpu"

    def seed(self, value: int) -> None:
        if self.torch is not None:
            self.torch.manual_seed(value)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(value)

    def reset_peak(self) -> None:
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def peak_mb(self) -> float | None:
        if self.torch is not None and self.torch.cuda.is_available():
            return round(self.torch.cuda.max_memory_allocated() / 2**20, 1)
        return None

    def sync(self) -> None:
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()


# ---- engines: load() -> synth(text, out_path, seed) -----------------------------------
def load_mock(ctx: dict[str, Any]):
    from scripts.mock_tts_engine import MockTTSEngine

    engine = MockTTSEngine()

    def synth(text: str, out_path: Path, seed: int) -> None:
        engine.tts_to_file(text=text, file_path=str(out_path))

    return synth, {"model": "mock-sine-v1"}


def load_xtts(ctx: dict[str, Any]):
    # Ürün kodundaki adaptör: XTTS çağrısı Studio/demo ile birebir aynı (varsayılan preset).
    from voxforge.engines import SynthesisRequest, create_engine

    engine = create_engine("xtts")
    reference = Path(ctx["reference"])

    def synth(text: str, out_path: Path, seed: int) -> None:
        engine.synthesize(SynthesisRequest(text=text, out_path=out_path, language="tr", reference_path=reference))

    return synth, {"model": engine.capabilities.model_id, "preset": "Dengeli (varsayılan)"}


def load_chatterbox(ctx: dict[str, Any]):
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    load_kwargs, dropped = supported_kwargs(
        ChatterboxMultilingualTTS.from_pretrained, {"device": ctx["device"], "t3_model": "v3"}
    )
    if "t3_model" in dropped:
        raise RuntimeError(
            "Kurulu chatterbox-tts sürümü t3_model='v3' desteklemiyor; V3 yerine V2 çalışırdı. "
            "Ortamı güncelleyin (setup_benchmark_env.ps1 -Engine chatterbox)."
        )
    model = ChatterboxMultilingualTTS.from_pretrained(**load_kwargs)
    generate_kwargs, generate_dropped = supported_kwargs(
        model.generate, {"language_id": "tr", "audio_prompt_path": str(ctx["reference"])}
    )
    if generate_dropped:
        raise RuntimeError(f"Chatterbox generate() beklenen argümanları kabul etmiyor: {generate_dropped}")

    def synth(text: str, out_path: Path, seed: int) -> None:
        wav = model.generate(text, **generate_kwargs)
        torchaudio.save(str(out_path), wav.detach().cpu(), model.sr)

    return synth, {
        "model": "ResembleAI/chatterbox t3_model=v3",
        "language_id": "tr",
        "sample_rate": model.sr,
        "note": "Çıktılar PerTh filigranı içerir (kütüphane varsayılanı).",
    }


def load_voxcpm(ctx: dict[str, Any], ultimate: bool):
    import soundfile as sf
    from voxcpm import VoxCPM

    load_kwargs, _ = supported_kwargs(VoxCPM.from_pretrained, {"load_denoiser": False})
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", **load_kwargs)
    wanted: dict[str, Any] = {"reference_wav_path": str(ctx["reference"]), "cfg_value": 2.0, "inference_timesteps": 10}
    if ultimate:
        transcript = ctx.get("reference_transcript")
        if not transcript:
            raise RuntimeError("voxcpm2-ultimate için referansın transkripti gerekir (prepare --reference-text).")
        wanted.update(prompt_wav_path=str(ctx["reference"]), prompt_text=transcript)
    generate_kwargs, dropped = supported_kwargs(model.generate, wanted)
    if "reference_wav_path" in dropped or (ultimate and "prompt_text" in dropped):
        raise RuntimeError(f"Kurulu voxcpm sürümü beklenen klonlama argümanlarını kabul etmiyor: {dropped}")
    sample_rate = model.tts_model.sample_rate

    def synth(text: str, out_path: Path, seed: int) -> None:
        kwargs = dict(generate_kwargs)
        if "seed" in inspect.signature(model.generate).parameters:
            kwargs["seed"] = seed
        wav = model.generate(text=text, **kwargs)
        sf.write(str(out_path), wav, sample_rate)

    return synth, {
        "model": "openbmb/VoxCPM2",
        "mode": "ultimate (referans + transkript)" if ultimate else "controllable (yalnız referans)",
        "sample_rate": sample_rate,
        "generate_kwargs": {k: v for k, v in generate_kwargs.items() if k not in {"prompt_text"}},
        "dropped_kwargs": dropped,
    }


LOADERS = {
    "mock": load_mock,
    "xtts": load_xtts,
    "chatterbox-v3": load_chatterbox,
    "voxcpm2": lambda ctx: load_voxcpm(ctx, ultimate=False),
    "voxcpm2-ultimate": lambda ctx: load_voxcpm(ctx, ultimate=True),
}


def run(run_dir: Path, engine: str, overwrite: bool, only: set[str] | None) -> int:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir = run_dir / engine
    results_path = out_dir / "results.json"
    if results_path.exists() and not overwrite:
        print(f"Zaten var: {results_path} (yeniden üretmek için --overwrite)", file=sys.stderr)
        return 2
    out_dir.mkdir(exist_ok=True)
    torch_helper = Torch()
    ctx = {
        "device": torch_helper.device,
        "reference": run_dir / "reference.wav",
        "reference_transcript": manifest["reference"].get("transcript"),
    }
    results: dict[str, Any] = {
        "engine": engine,
        "run_id": manifest["run_id"],
        "testset_version": manifest["testset_version"],
        "status": "running",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "device": ctx["device"],
        "versions": package_versions(),
        "settings": None,
        "load_seconds": None,
        "vram_peak_mb": None,
        "items": [],
        "error": None,
    }
    print(f"Motor: {engine} · cihaz: {ctx['device']} · çıktı: {out_dir}")
    try:
        torch_helper.reset_peak()
        started = time.perf_counter()
        synth, settings = LOADERS[engine](ctx)
        torch_helper.sync()
        results["load_seconds"] = round(time.perf_counter() - started, 2)
        results["settings"] = settings
    except Exception as exc:  # noqa: BLE001 - recorded, run marked failed
        results.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utc_now_iso())
        results["traceback"] = traceback.format_exc(limit=5)
        write_json_atomic(results_path, results)
        print(f"HATA (model yüklenemedi): {results['error']}", file=sys.stderr)
        return 1

    for item in manifest["items"]:
        if only and item["item_id"] not in only:
            continue
        out_path = out_dir / f"{item['item_id']}.wav"
        entry: dict[str, Any] = {"item_id": item["item_id"], "path": str(out_path), "ok": False, "error": None}
        try:
            torch_helper.seed(manifest["seed"])
            began = time.perf_counter()
            synth(item["spoken"], out_path, manifest["seed"])
            torch_helper.sync()
            entry["seconds"] = round(time.perf_counter() - began, 3)
            entry["audio_seconds"] = round(wav_seconds(out_path), 3)
            entry["rtf"] = round(entry["seconds"] / entry["audio_seconds"], 3) if entry["audio_seconds"] else None
            entry["ok"] = entry["audio_seconds"] > 0
            if not entry["ok"]:
                entry["error"] = "boş ses"
        except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM; continue with next item
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if torch_helper.torch is not None and torch_helper.torch.cuda.is_available():
                torch_helper.torch.cuda.empty_cache()
        results["items"].append(entry)
        state = "ok" if entry["ok"] else f"HATA {entry['error']}"
        print(f"  {item['item_id']}: {state} {entry.get('seconds', '')}s")
        write_json_atomic(results_path, {**results, "vram_peak_mb": torch_helper.peak_mb()})

    results.update(status="completed", finished_at=utc_now_iso(), vram_peak_mb=torch_helper.peak_mb())
    write_json_atomic(results_path, results)
    failed = sum(not entry["ok"] for entry in results["items"])
    print(f"Bitti: {len(results['items']) - failed} başarılı, {failed} hatalı. VRAM tepe: {results['vram_peak_mb']} MB")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="prepare adımının oluşturduğu klasör")
    parser.add_argument("--engine", required=True, choices=ENGINE_CHOICES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", default="", help="Yalnız bu maddeler (ör. t01,t03) — hızlı deneme için")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    only = {part.strip() for part in args.only.split(",") if part.strip()} or None
    return run(args.run.resolve(), args.engine, args.overwrite, only)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
