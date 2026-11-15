# -*- coding: utf-8 -*-
"""VF-018 limited real-GPU validation (main .venv; run with no other Playground open).

    .venv\\Scripts\\python.exe scripts/quality/vf018_validate.py [--profile mimo] [--baseline <record id>]

1. Short-text regression: a fresh VoxCPM2 worker, default settings, seed 1234, the text of
   ``--baseline`` (a first-in-process generation) -> the WAV must be byte-identical.
2. Quality Director on real outputs (faster-whisper large-v3 on CPU) while VoxCPM2 stays
   loaded on the GPU; GPU memory is sampled to show the ASR does not use VRAM.
3. Real-audio fault injection: the same WAV cut short and with a repeated piece.
4. Two alternatives with new seeds + selection/undo; then both workers are closed.
Evidence goes to ``outputs/vf018/`` (git ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.quality import PlaygroundQuality  # noqa: E402
from voxforge.playground.service import PlaygroundService  # noqa: E402
from voxforge.quality.checks import summarize  # noqa: E402
from voxforge.runs import write_json_atomic  # noqa: E402

OUT = REPO_ROOT / "outputs" / "vf018"
DEFAULT_BASELINE = "20261107-135042_956800_voxcpm2"
NARRATION = (
    "Krallığın kuzeyinde, yüzyıllardır kimsenin yaklaşmaya cesaret edemediği eski bir kale vardı. "
    "Fakat o gece, nöbetçiler içeriden gelen ayak seslerini duydu."
)


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class GpuSampler:
    def __init__(self) -> None:
        self.values: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = gpu_used_mb()
            if value is not None:
                self.values.append(value)
            self._stop.wait(0.5)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        return {"min_mb": min(self.values, default=None), "max_mb": max(self.values, default=None), "samples": len(self.values)}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edit_wav(source: Path, target: Path, mode: str) -> Path:
    with wave.open(str(source), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    width = params.sampwidth * params.nchannels
    count = len(frames) // width
    if mode == "cut":  # konuşmanın ortasında kes (son kelimeler yok, sessizliksiz bitiş)
        data = frames[: int(count * 0.55) * width]
    elif mode == "repeat":  # orta bölümü bir kez daha söylet
        a, b = int(count * 0.35) * width, int(count * 0.6) * width
        data = frames[:b] + frames[a:b] + frames[b:]
    else:
        raise ValueError(mode)
    with wave.open(str(target), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(data)
    return target


def brief(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": report["verdict"],
        "summary": summarize(report),
        "asr_text": (report.get("asr") or {}).get("text"),
        "asr_seconds": (report.get("asr") or {}).get("seconds"),
        "asr_load_seconds": (report.get("asr") or {}).get("load_seconds"),
        "check_seconds": report.get("check_seconds"),
        "metrics": {k: report["metrics"].get(k) for k in ("cer", "rate", "seconds_per_char", "longest_pause", "mean_dbfs", "asr_mean_prob", "similarity")},
        "findings": [(f["code"], f["severity"], f["confidence"], f["source"], f["message"]) for f in report["findings"]],
    }


def recheck() -> int:
    """Same WAVs as the last validation, current checker version, CPU ASR only (no TTS)."""
    from voxforge.quality.checks import CHECKER_VERSION

    path = OUT / "validation.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    service = PlaygroundService()
    quality = PlaygroundQuality(service)
    rows: dict[str, Any] = {}
    try:
        items = [("short", result["regression"]["new_record"]), ("narration", result["check_narration"]["record"])]
        for name, record_id in items:
            rows[name] = brief(quality.check(record_id, force=True))
            print(name, rows[name]["summary"], flush=True)
        narration = service.get(result["check_narration"]["record"])
        for mode in ("cut", "repeat"):
            wav = OUT / f"{narration['id']}_{mode}.wav"
            rows[f"injected_{mode}"] = brief(quality.director.check(wav, narration["spoken_text"], None, Path(narration["reference"])))
            print(mode, rows[f"injected_{mode}"]["summary"], flush=True)
        group = quality.store.load(result["takes"]["group"])
        for take in group["takes"]:
            if take["take_id"] == "t1":
                continue
            fresh = quality.takes.check_take(group, take)
            quality.store.update_take(group["group_id"], take["take_id"], **fresh)
            rows[take["take_id"]] = brief(fresh["quality"])
            print(take["take_id"], rows[take["take_id"]]["summary"], flush=True)
        orig = quality.takes.check_take(group, group["takes"][0])
        quality.store.update_take(group["group_id"], "t1", **orig)
        rows["recommended"] = quality.store.load(group["group_id"])["recommended"]
    finally:
        quality.close()
    result[f"recheck_{CHECKER_VERSION}"] = rows
    write_json_atomic(path, result)
    print("yazıldı:", path)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="mimo")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--alternatives", type=int, default=2)
    parser.add_argument("--recheck", action="store_true", help="GPU'suz: önceki doğrulamanın seslerini güncel kurallarla yeniden denetle")
    args = parser.parse_args(argv)
    if args.recheck:
        return recheck()

    service = PlaygroundService()
    quality = PlaygroundQuality(service)
    result: dict[str, Any] = {"gpu_idle_mb": gpu_used_mb()}
    try:
        baseline = service.get(args.baseline)
        if baseline is None:
            raise SystemExit(f"Temel kayıt yok: {args.baseline}")
        # 1) kısa metin regresyonu: yeni süreç, ilk üretim
        service.manager.unload()
        record = service.generate(baseline["text"], "voxcpm2", args.profile, seed=1234)
        result["regression"] = {
            "baseline": args.baseline,
            "baseline_sha256": sha(baseline["wav"]),
            "new_record": record["id"],
            "new_sha256": sha(record["wav"]),
            "identical": sha(baseline["wav"]) == sha(record["wav"]),
            "generation_seconds": record["timing"]["generation_seconds"],
            "model_load_seconds": record["timing"]["model_load_seconds"],
            "vram_peak_mb": record["vram"].get("generation_peak_mb"),
            "gpu_used_after_tts_mb": gpu_used_mb(),
        }
        print("regresyon:", result["regression"]["identical"], flush=True)

        # 2) gerçek çıktılar üzerinde denetim (VoxCPM2 GPU'da yüklü kalır)
        with GpuSampler() as sampler:
            report = quality.check(record["id"], force=True)
        result["check_short"] = {**brief(report), "gpu_during_asr": sampler.summary(), "tts_still_loaded": service.manager.current_engine}
        print("kısa:", summarize(report), flush=True)
        narration = service.generate(NARRATION, "voxcpm2", args.profile, seed=1234)
        report = quality.check(narration["id"], force=True)
        result["check_narration"] = {"record": narration["id"], **brief(report)}
        print("anlatım:", summarize(report), flush=True)

        # 3) gerçek sesle hata enjeksiyonu
        OUT.mkdir(parents=True, exist_ok=True)
        for mode in ("cut", "repeat"):
            wav = edit_wav(Path(narration["wav"]), OUT / f"{narration['id']}_{mode}.wav", mode)
            report = quality.director.check(wav, narration["spoken_text"], None, Path(narration["reference"]))
            result[f"injected_{mode}"] = brief(report)
            print(mode, summarize(report), flush=True)

        # 4) alternatifler, seçim ve geri dönüş
        started = time.perf_counter()
        with GpuSampler() as sampler:
            group, errors = quality.alternatives(record["id"], args.alternatives)
        result["takes"] = {
            "group": group["group_id"],
            "seconds": round(time.perf_counter() - started, 1),
            "errors": errors,
            "gpu": sampler.summary(),
            "recommended": group["recommended"],
            "takes": [
                {"take": t["take_id"], "seed": t["seed"], "summary": summarize(t.get("quality")), "similarity": t.get("similarity"),
                 "rtf": (t.get("timing") or {}).get("rtf"), "vram_peak_mb": t.get("vram_peak_mb")}
                for t in group["takes"]
            ],
        }
        other = next(t["take_id"] for t in group["takes"] if t["take_id"] != "t1")
        quality.select(group["group_id"], other)
        reverted = quality.revert(group["group_id"])
        result["takes"]["undo"] = {"selected_then": other, "after_revert": reverted["selected"]}
        print("take'ler:", json.dumps(result["takes"], ensure_ascii=False)[:400], flush=True)
    finally:
        asr_proc = quality.asr.manager._current.process if quality.asr.manager._current else None
        tts_proc = service.manager._current.process if service.manager._current else None
        quality.close()
        service.manager.close()
        time.sleep(1.0)
        result["lifecycle"] = {
            "asr_exited": asr_proc is None or asr_proc.poll() is not None,
            "tts_exited": tts_proc is None or tts_proc.poll() is not None,
            "gpu_after_close_mb": gpu_used_mb(),
        }
        write_json_atomic(OUT / "validation.json", result)
        print("yazıldı:", OUT / "validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
