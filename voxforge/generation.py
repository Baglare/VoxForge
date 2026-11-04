"""Engine-agnostic text-to-WAV generation (VF-003).

Moved out of ``app/gradio_xtts_demo.py`` so it can be tested with the Mock
engine and reused by the future Studio queue. Behaviour is unchanged: chunk
long text, synthesize every part, validate parts, concatenate, optionally
normalize with FFmpeg loudnorm and keep the raw file if that fails.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from scripts.audio_concat_utils import concatenate_wavs
from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks
from voxforge.engines.base import SynthesisRequest, TTSEngine
from voxforge.runs import wav_part_status

DEFAULT_MAX_CHARS = 220
OUTPUT_NORMALIZE_FILTER = "loudnorm=I=-18:TP=-2:LRA=11"


def raise_on_failed_parts(parts: list[dict[str, Any]]) -> None:
    """Eksik/bos/okunamayan parca varken basari raporlanmaz."""
    failed = [part for part in parts if not part["ok"]]
    if failed:
        details = "; ".join(f"parca {part['index']}: {part['error']}" for part in failed)
        raise RuntimeError(f"Uretilen ses parcalari gecersiz: {details}")


def normalize_wav(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Final WAV icin hafif FFmpeg loudnorm post-processing uygular."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, "FFmpeg bulunamadı; ham çıktı korundu."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        OUTPUT_NORMALIZE_FILTER,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    print("Cikti post-processing FFmpeg komutu:")
    print(subprocess.list2cmdline(command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "bilinmeyen FFmpeg hatasi"
        return False, f"Çıktı normalize edilemedi; ham çıktı korundu. Ayrıntı: {detail}"

    if not output_path.is_file():
        return False, "Normalize komutu tamamlandı ancak işlenmiş çıktı oluşmadı; ham çıktı korundu."

    return True, "Çıktı sesi normalize edildi."


def synthesize_text(
    engine: TTSEngine,
    text: str,
    *,
    output_dir: Path,
    output_stem: str,
    reference_path: Path | None,
    preset: dict[str, Any] | None = None,
    preset_label: str = "",
    post_processing_enabled: bool = False,
    language: str = "tr",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Tek veya chunked uretim yapar; final cikti yolunu ve parca durumlarini dondurur."""
    chunks = split_text_for_tts(text, max_chars=max_chars) if len(text) > max_chars else [text]
    chunk_summary = summarize_chunks(chunks)
    raw_output_path = output_dir / f"{output_stem}_raw.wav"
    warnings: list[str] = []
    parts: list[dict[str, Any]] = []
    applied_params: dict[str, Any] = {}

    def synth(chunk_text: str, out_path: Path) -> None:
        nonlocal applied_params
        result = engine.synthesize(
            SynthesisRequest(
                text=chunk_text,
                out_path=out_path,
                language=language,
                reference_path=reference_path,
                preset=dict(preset or {}),
            )
        )
        applied_params = result.params
        for warning in result.warnings:
            if warning not in warnings:
                warnings.append(warning)

    if chunk_summary["chunking_used"]:
        chunk_dir = output_dir / "chunks" / output_stem
        chunk_paths: list[Path] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"chunk_{index:02d}.wav"
            synth(chunk, chunk_path)
            chunk_paths.append(chunk_path)
            parts.append({"index": index, **wav_part_status(chunk_path)})

        raise_on_failed_parts(parts)
        concat_ok, concat_message = concatenate_wavs(chunk_paths, raw_output_path)
        if not concat_ok:
            raise RuntimeError(f"WAV parçaları birleştirilemedi: {concat_message}")
    else:
        synth(chunks[0], raw_output_path)
        parts.append({"index": 1, **wav_part_status(raw_output_path)})
        raise_on_failed_parts(parts)

    final_output_path = raw_output_path
    post_processing_applied = False
    if post_processing_enabled:
        normalized_output_path = output_dir / f"{output_stem}_normalized.wav"
        normalized_ok, normalized_message = normalize_wav(raw_output_path, normalized_output_path)
        if normalized_ok:
            final_output_path = normalized_output_path
            post_processing_applied = True
        else:
            warnings.append(normalized_message)

    caps = engine.capabilities
    return {
        "preset_label": preset_label,
        "engine_id": caps.engine_id,
        "model_id": caps.model_id,
        "applied_params": applied_params,
        "raw_output_path": raw_output_path,
        "final_output_path": final_output_path,
        "post_processing_enabled": post_processing_enabled,
        "post_processing_applied": post_processing_applied,
        "chunking_used": chunk_summary["chunking_used"],
        "chunk_count": chunk_summary["chunk_count"],
        "chunks": chunk_summary["chunks"],
        "parts": parts,
        "warnings": warnings,
    }
