# -*- coding: utf-8 -*-
"""VF-022 paired emotion pilot. Use only existing local voice data and model caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voxforge.benchmark.metrics import score_transcript  # noqa: E402
from voxforge.playground.rhythm import analyze  # noqa: E402
from voxforge.playground.workers import EngineManager, worker_env  # noqa: E402

SEED = 1234
SETTINGS = {"cfg_value": 2.0, "inference_timesteps": 10}
CASES = (
    ("sevinç", "Sonunda kapı açıldı. İçeri girip pencerenin yanına yürüdüm.",
     "Sevinçli ve içten bir sesle konuş.", "Speak in a joyful and heartfelt voice."),
    ("kaygı", "Şimdi koridorun sonuna kadar gidip orada bekleyeceğiz.",
     "Kaygılı ve tedirgin bir sesle konuş.", "Speak in an anxious and uneasy voice."),
    ("üzüntü", "Masada duran defteri kapattım. Bir süre öylece bekledim.",
     "Üzgün ve kederli bir sesle konuş.", "Speak in a sad and sorrowful voice."),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_environment() -> dict[str, str]:
    env = worker_env()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def synthesize(out: Path, reference: Path) -> None:
    if not reference.is_file():
        raise FileNotFoundError(reference)
    out.mkdir(parents=True)
    blind = out / "dinleme"
    blind.mkdir()
    anchor = blind / "kimlik-referansi.wav"
    shutil.copyfile(reference, anchor)
    # Rotate one masked permutation so each condition occupies each position once.
    order = ["stilsiz", "tr", "en"]
    random.Random(SEED).shuffle(order)
    rows = []
    manager = EngineManager(out / "logs", env=local_environment())
    try:
        for line_no, (emotion, text, turkish, english) in enumerate(CASES, start=1):
            styles = {"stilsiz": None, "tr": turkish, "en": english}
            for position, condition in enumerate(order, start=1):
                clip = f"R{line_no}-{position}"
                wav = blind / f"{clip}.wav"
                style = styles[condition]
                reply = manager.synthesize("voxcpm2", {
                    "reference": str(anchor), "chunks": [text], "style": style,
                    "seed": SEED, "settings": SETTINGS, "out": str(wav),
                })
                applied = dict(reply["applied_settings"])
                if (
                    applied.pop("style", None) != style
                    or applied.pop("mode") != "controllable (yalnız referans)"
                    or applied != SETTINGS
                    or reply["seed"] != SEED
                    or reply["chunks"] != 1
                    or reply["join"] != {"pauses": None, "trim": False}
                ):
                    raise RuntimeError(f"{clip}: worker beklenen deney ayarlarını uygulamadı.")
                with wave.open(str(wav), "rb") as audio:
                    if not audio.getnframes():
                        raise RuntimeError(f"{clip}: boş WAV.")
                    sample_rate = audio.getframerate()
                rows.append({
                    "clip": clip, "wav": wav.relative_to(out).as_posix(), "sha256": digest(wav),
                    "line": line_no, "text": text, "emotion": emotion, "condition": condition,
                    "style": style, "sample_rate": sample_rate, "seconds": reply["audio_seconds"],
                    "rtf": reply["rtf"], "peak_allocated_mib": reply["vram_peak_mb"],
                })
                print(f"{clip}: {reply['audio_seconds']} s", flush=True)
            order = order[1:] + order[:1]
        worker = reply["worker"]
        write_json(out / "anahtar.json", {
            "engine": "voxcpm2", "model_revision": worker["source"]["revision"],
            "reference_sha256": digest(anchor), "seed": SEED, "settings": SETTINGS,
            "mode": reply["applied_settings"]["mode"], "join": reply["join"],
            "device": worker["device"], "compile": worker["compile"], "clips": rows,
        })
    finally:
        manager.close()

    with (blind / "puanlama.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["klip", "algılanan_duygu", "duygu_belirginliği_1_5", "doğallık_1_5",
                         "kimlik_benzerliği_1_5", "başka_biri_gibi_evet_hayır", "telaffuz_ritim_notu"])
        writer.writerows([[row["clip"], "", "", "", "", "", ""] for row in rows])
    (blind / "DINLEME.md").write_text(
        "# VF-022 kör dinleme\n\n"
        "R1, R2 ve R3 üç ayrı replik. Her grubun üç okuması var. Önce kimlik referansını dinle. "
        "Klipleri dosya numarasına göre, aynı ses seviyesinde dinle ve her klibi ayrı puanla.\n\n"
        "Algıladığın duyguyu kendi sözcüklerinle yaz. Nötr veya belirsiz seçilebilir. "
        "Belirginlikte 1 = anlaşılmıyor, 5 = çok belirgin. Doğallıkta 1 = yapay, 5 = doğal. "
        "Kimlikte 1 = başka konuşmacı, 5 = referansla aynı konuşmacı. "
        "Kimlik değişimi duyduysan ilgili sütuna evet yaz. Telaffuz, vurgu ve duraklama notlarını ayrıca ekle.\n\n"
        "Puanları kaydettikten sonra anahtar açılır. Duygu eşleşmesi ve belirginliği, doğallık ve kimlik ayrı okunur. "
        "Başka konuşmacı gibi duyulan örnek, duygusu daha belirgin olsa da iyileşme sayılmaz.\n",
        encoding="utf-8",
    )
    shutil.make_archive(str(out / "kor-dinleme"), "zip", blind)


def measure(out: Path) -> None:
    os.environ.update(local_environment())
    from vf017_analyze import Evaluator, f0_track, load, semitones

    key = json.loads((out / "anahtar.json").read_text(encoding="utf-8"))
    anchor = out / "dinleme" / "kimlik-referansi.wav"
    if digest(anchor) != key["reference_sha256"]:
        raise ValueError("Kimlik referansı değişmiş.")
    rows = key["clips"]
    for row in rows:
        if digest(out / row["wav"]) != row["sha256"]:
            raise ValueError(f"{row['clip']}: ses dosyası değişmiş.")
    evaluator = Evaluator()
    reference = evaluator.embedding(anchor)
    embeddings = {row["clip"]: evaluator.embedding(out / row["wav"]) for row in rows}
    baselines = {row["line"]: embeddings[row["clip"]] for row in rows if row["condition"] == "stilsiz"}
    results = []
    for row in rows:
        path = out / row["wav"]
        words, transcript = evaluator.words(path)
        samples, rate = load(path)
        embedding = embeddings[row["clip"]]
        results.append({
            **row,
            "asr_text": transcript,
            **score_transcript(row["text"], transcript),
            "ecapa_reference": evaluator.cosine(reference, embedding),
            "ecapa_baseline": evaluator.cosine(baselines[row["line"]], embedding),
            "f0_range_st": semitones(f0_track(samples, rate)),
            **analyze(path, row["text"], words),
        })
        print(f"{row['clip']}: ölçüldü", flush=True)
    write_json(out / "olcumler.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["synth", "measure"])
    parser.add_argument("--out", required=True, type=Path, help="Git dışında, özel çıktı dizini")
    parser.add_argument("--reference", type=Path, help="Mevcut kayıtlı referans (synth için gerekli)")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "synth":
        if args.reference is None:
            parser.error("synth için --reference gerekli.")
        synthesize(out, args.reference.resolve())
    else:
        measure(out)


if __name__ == "__main__":
    main()
