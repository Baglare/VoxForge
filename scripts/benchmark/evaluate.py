# -*- coding: utf-8 -*-
"""Objective evaluation of benchmark outputs, inside the separate eval venv (VF-Q1).

    <eval-venv>\\python.exe scripts/benchmark/evaluate.py --run <run_dir>

For every engine with results.json it writes evaluation/<engine>.json containing
- the faster-whisper (large-v3, language=tr) transcript of each output, used for
  CER and insertion ("metin dışı ses") metrics in the report;
- cosine speaker similarity between the run's reference and each output
  (SpeechBrain ECAPA-TDNN, 16 kHz).
Models are loaded once and run on the GPU if available. The same ASR/similarity
models score every engine, so the comparison is fair even though they are only
proxies for human judgement.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.runs import utc_now_iso, write_json_atomic  # noqa: E402

ASR_MODEL = "large-v3"
SIM_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class Evaluator:
    def __init__(self, models_dir: Path, use_asr: bool, use_sim: bool) -> None:
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr = None
        self.encoder = None
        if use_asr:
            from faster_whisper import WhisperModel

            compute = "float16" if self.device == "cuda" else "int8"
            self.asr = WhisperModel(ASR_MODEL, device=self.device, compute_type=compute)
        if use_sim:
            from speechbrain.inference.speaker import EncoderClassifier

            kwargs: dict[str, Any] = {
                "source": SIM_MODEL,
                "savedir": str(models_dir / "spkrec-ecapa-voxceleb"),
                "run_opts": {"device": self.device},
            }
            try:  # Windows: sembolik link yetkisi gerektirmesin
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except ImportError:
                pass
            self.encoder = EncoderClassifier.from_hparams(**kwargs)

    def transcribe(self, path: Path) -> str:
        segments, _info = self.asr.transcribe(str(path), language="tr", beam_size=5, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()

    def embedding(self, path: Path):
        import soundfile as sf
        import torchaudio.functional as F

        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = self.torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        if rate != 16000:
            tensor = F.resample(tensor, rate, 16000)
        with self.torch.no_grad():
            return self.encoder.encode_batch(tensor.to(self.device)).squeeze()

    def similarity(self, reference_embedding, path: Path) -> float:
        other = self.embedding(path)
        return round(float(self.torch.nn.functional.cosine_similarity(reference_embedding, other, dim=0)), 4)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--engines", default="", help="Virgülle (varsayılan: sonucu olan tüm motorlar)")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=None, help="SpeechBrain model klasörü")
    args = parser.parse_args(argv)

    run_dir = args.run.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = {name.strip() for name in args.engines.split(",") if name.strip()}
    engines = [
        path.parent.name
        for path in sorted(run_dir.glob("*/results.json"))
        if not wanted or path.parent.name in wanted
    ]
    if not engines:
        print("Değerlendirilecek motor sonucu yok.", file=sys.stderr)
        return 1
    models_dir = args.models_dir or (run_dir.parent / "_eval_models")
    evaluator = Evaluator(models_dir, use_asr=not args.no_asr, use_sim=not args.no_sim)
    reference_embedding = evaluator.embedding(run_dir / "reference.wav") if evaluator.encoder else None

    for engine in engines:
        results = json.loads((run_dir / engine / "results.json").read_text(encoding="utf-8"))
        output = {
            "engine": engine,
            "run_id": manifest["run_id"],
            "evaluated_at": utc_now_iso(),
            "device": evaluator.device,
            "asr_model": f"faster-whisper {ASR_MODEL}" if evaluator.asr else None,
            "sim_model": SIM_MODEL if evaluator.encoder else None,
            "versions": {name: version(name) for name in ("faster-whisper", "speechbrain", "torch")},
            "items": [],
        }
        for entry in results.get("items", []):
            if not entry.get("ok"):
                continue
            path = Path(entry["path"])
            row: dict[str, Any] = {"item_id": entry["item_id"]}
            try:
                if evaluator.asr:
                    row["transcript"] = evaluator.transcribe(path)
                if reference_embedding is not None:
                    row["similarity"] = evaluator.similarity(reference_embedding, path)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            output["items"].append(row)
            print(f"{engine} {entry['item_id']}: {row.get('transcript', '')[:60]} | sim={row.get('similarity')}")
        write_json_atomic(run_dir / "evaluation" / f"{engine}.json", output)
    print("Değerlendirme yazıldı:", run_dir / "evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
