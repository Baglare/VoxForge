# -*- coding: utf-8 -*-
"""Quality Director ASR worker (VF-018): faster-whisper words + optional speaker similarity.

Runs inside the existing eval environment (``VoxForge-bench-eval``: faster-whisper
large-v3 and SpeechBrain ECAPA are already cached for VF-Q1, nothing is downloaded).
Started by ``voxforge.quality.asr.AsrClient`` through the same ``EngineManager`` protocol
as the TTS workers: JSON lines on stdin, replies on the original stdout prefixed with
``@@VF ``; library output goes to stderr.

    -> {"cmd": "transcribe", "id": "...", "wav": "x.wav", "reference": "ref.wav" | null}
    <- {"event": "result", "id": "...", "ok": true, "text": "...", "words": [...],
        "seconds": ..., "similarity": 0.81 | null}
    -> {"cmd": "shutdown"}

Default device is CPU (int8): next to a loaded VoxCPM2 (~5.8 GB hot + desktop apps) the
8 GB laptop GPU has no room for large-v3, and the manager hides the GPU from CPU workers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROTOCOL_PREFIX = "@@VF "
ASR_MODEL = os.environ.get("VOXFORGE_QUALITY_ASR_MODEL", "large-v3")
SIM_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class Channel:
    def __init__(self) -> None:
        sys.stdout.flush()
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr

    def send(self, payload: dict[str, Any]) -> None:
        self._out.write(PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
        self._out.flush()


class Recogniser:
    def __init__(self, device: str, beam_size: int) -> None:
        from faster_whisper import WhisperModel

        self.device = device
        self.beam_size = beam_size
        threads = int(os.environ.get("VOXFORGE_QUALITY_ASR_THREADS", "0")) or min(16, os.cpu_count() or 4)
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(ASR_MODEL, device=device, compute_type=compute, cpu_threads=threads)
        self.info = {"model": f"faster-whisper {ASR_MODEL}", "compute_type": compute, "cpu_threads": threads, "beam_size": beam_size}
        self._encoder = None
        self._embeddings: dict[tuple[str, int, float], Any] = {}

    def transcribe(self, wav: str) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            wav, language="tr", beam_size=self.beam_size, vad_filter=False, word_timestamps=True, condition_on_previous_text=False
        )
        words, text, segment_info = [], [], []
        for segment in segments:
            text.append(segment.text)
            segment_info.append({"avg_logprob": round(segment.avg_logprob, 3), "no_speech_prob": round(segment.no_speech_prob, 3)})
            for word in segment.words or []:
                words.append({"word": word.word.strip(), "start": round(word.start, 3), "end": round(word.end, 3), "prob": round(word.probability, 3)})
        return {"text": "".join(text).strip(), "words": words, "segments": segment_info, "duration": round(info.duration, 3)}

    # -- speaker similarity (ECAPA, same model as the VF-Q1 evaluation) ------------------------
    def _embedding(self, path: str):
        import soundfile as sf
        import torch
        import torchaudio.functional as F

        stat = Path(path).stat()
        key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime)
        if key in self._embeddings:
            return self._embeddings[key]
        if self._encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier

            kwargs: dict[str, Any] = {
                "source": SIM_MODEL,
                "savedir": str(_sim_dir()),
                "run_opts": {"device": self.device},
            }
            try:
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except ImportError:
                pass
            self._encoder = EncoderClassifier.from_hparams(**kwargs)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        if rate != 16000:
            tensor = F.resample(tensor, rate, 16000)
        with torch.no_grad():
            embedding = self._encoder.encode_batch(tensor.to(self.device)).squeeze()
        if len(self._embeddings) > 64:
            self._embeddings.clear()
        self._embeddings[key] = embedding
        return embedding

    def similarity(self, reference: str, wav: str) -> float:
        import torch

        a, b = self._embedding(reference), self._embedding(wav)
        return round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 4)


def _sim_dir() -> Path:
    """SpeechBrain copy already used by VF-017 (``HF_HOME/spkrec-ecapa-voxceleb``); no new download."""
    configured = os.environ.get("VOXFORGE_QUALITY_SIM_DIR")
    if configured:
        return Path(configured)
    return Path(os.environ.get("HF_HOME", "C:/AvenoxData/Models/HF")) / "spkrec-ecapa-voxceleb"


def serve(channel: Channel, device: str, beam_size: int) -> int:
    try:
        started = time.perf_counter()
        recogniser = Recogniser(device, beam_size)
        load_seconds = round(time.perf_counter() - started, 2)
    except Exception as exc:  # noqa: BLE001
        channel.send({"event": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
        return 1
    channel.send({"event": "ready", "engine": "asr", "device": device, "load_seconds": load_seconds, "model": recogniser.info["model"], **recogniser.info})
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
        if request.get("cmd") != "transcribe":
            channel.send({"event": "result", "id": request.get("id"), "ok": False, "error": "Bilinmeyen komut"})
            continue
        try:
            started = time.perf_counter()
            reply: dict[str, Any] = {"ok": True, **recogniser.transcribe(request["wav"])}
            reply["seconds"] = round(time.perf_counter() - started, 2)
            reply["similarity"] = None
            if request.get("reference"):
                try:
                    reply["similarity"] = recogniser.similarity(request["reference"], request["wav"])
                except Exception as exc:  # noqa: BLE001 - benzerlik isteğe bağlı, transkripti düşürmez
                    reply["similarity_error"] = f"{type(exc).__name__}: {exc}"
            reply["asr"] = recogniser.info
        except Exception as exc:  # noqa: BLE001
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)}
        channel.send({"event": "result", "id": request.get("id"), **reply})
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="asr")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--beam-size", type=int, default=int(os.environ.get("VOXFORGE_QUALITY_ASR_BEAM", "5")))
    args = parser.parse_args(argv)
    return serve(Channel(), args.device, args.beam_size)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
