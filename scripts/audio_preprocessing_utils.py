# -*- coding: utf-8 -*-
"""Shared safe reference-audio preprocessing helpers for VoxForge."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from scripts.audio_quality_utils import analyze_audio_file


MIN_SELECTED_SECONDS = 15.0
MIN_ORIGINAL_DURATION_RATIO = 0.70

# Safe mode intentionally avoids silence trimming. The goal is format stability,
# not shortening the speaker reference.
SAFE_AUDIO_FILTER = "loudnorm=I=-20:TP=-2:LRA=11"

# Available for explicit non-default use only. It is guarded by duration checks.
CONSERVATIVE_TRIM_AUDIO_FILTER = (
    "silenceremove="
    "start_periods=1:start_duration=0.50:start_threshold=-50dB:"
    "stop_periods=1:stop_duration=0.50:stop_threshold=-50dB,"
    f"{SAFE_AUDIO_FILTER}"
)

FALLBACK_WARNING = (
    "On isleme kaliteyi artirmadi, guvenli normalize edilmis referans kullanildi."
)


class PreprocessingError(Exception):
    """Raised when preprocessing cannot produce a safe reference file."""


def build_ffmpeg_command(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    audio_filter: str,
) -> list[str]:
    """Build a deterministic FFmpeg command for XTTS reference audio."""
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        audio_filter,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def run_ffmpeg_preprocess(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    audio_filter: str,
) -> None:
    """Run FFmpeg and fail with a clear local error message."""
    command = build_ffmpeg_command(ffmpeg_path, input_path, output_path, audio_filter)
    print("Referans ses on isleme FFmpeg komutu:")
    print(subprocess.list2cmdline(command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        detail = result.stderr.strip() or "FFmpeg hata ayrintisi dondurmedi."
        raise PreprocessingError(
            "FFmpeg referans ses on isleme adimi basarisiz oldu.\n"
            f"Ayrinti: {detail}"
        )

    if not output_path.is_file():
        raise PreprocessingError(
            "FFmpeg tamamlandi ancak on islenmis referans ses dosyasi olusmadi."
        )


def duration_rejection_reason(
    original_duration_seconds: float | None,
    candidate_duration_seconds: float | None,
) -> str | None:
    """Return why a candidate is too short to use, or None if it is safe."""
    if candidate_duration_seconds is None:
        return "On islenmis referans ses suresi okunamadi; cikti kullanilmadi."

    if candidate_duration_seconds < MIN_SELECTED_SECONDS:
        return (
            "On islenmis referans ses 15 saniyenin altina dustu; "
            "cikti kullanilmadi."
        )

    if original_duration_seconds and original_duration_seconds > 0:
        min_allowed = original_duration_seconds * MIN_ORIGINAL_DURATION_RATIO
        if candidate_duration_seconds < min_allowed:
            return (
                "On islenmis referans ses ham surenin %70'inden kisa; "
                "cikti kullanilmadi."
            )

    return None


def analyze_candidate(audio_path: Path, label: str) -> dict[str, Any]:
    """Run the shared quality analyzer with a small error wrapper."""
    try:
        return analyze_audio_file(audio_path)
    except Exception as exc:
        raise PreprocessingError(f"{label} kalite analizi basarisiz oldu: {exc}") from exc


def candidate_is_usable(
    original_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate duration and avoid worsening a non-BAD original into BAD."""
    duration_warning = duration_rejection_reason(
        original_report.get("duration_seconds"),
        candidate_report.get("duration_seconds"),
    )
    if duration_warning:
        return False, duration_warning

    original_quality = original_report.get("quality")
    candidate_quality = candidate_report.get("quality")
    if original_quality != "BAD" and candidate_quality == "BAD":
        return False, FALLBACK_WARNING

    return True, None


def preprocess_reference_audio(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "safe",
) -> dict[str, Any]:
    """Preprocess a reference voice file without allowing unsafe shortening.

    Returns a dict with the selected output, selected variant, duration details,
    warning text, and the quality reports for evaluated candidates.
    """
    source_path = Path(input_path)
    target_path = Path(output_path)

    if mode not in {"safe", "conservative_trim"}:
        raise PreprocessingError(f"Bilinmeyen on isleme modu: {mode}")

    if not source_path.is_file():
        raise PreprocessingError(f"Giris ses dosyasi bulunamadi: {source_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise PreprocessingError(
            "FFmpeg bulunamadi. Lutfen FFmpeg kurulumunu ve PATH ayarini kontrol edin."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    original_report = analyze_candidate(source_path, "Ham referans")
    original_duration = original_report.get("duration_seconds")
    candidate_reports: dict[str, Any] = {"original": original_report}
    preprocessing_warning: str | None = None

    if mode == "safe":
        run_ffmpeg_preprocess(
            ffmpeg_path,
            source_path,
            target_path,
            SAFE_AUDIO_FILTER,
        )
        safe_report = analyze_candidate(target_path, "Guvenli normalize referans")
        candidate_reports["safe_normalized"] = safe_report

        duration_warning = duration_rejection_reason(
            original_duration,
            safe_report.get("duration_seconds"),
        )
        if duration_warning:
            target_path.unlink(missing_ok=True)
            raise PreprocessingError(duration_warning)

        if original_report.get("quality") != "BAD" and safe_report.get("quality") == "BAD":
            preprocessing_warning = FALLBACK_WARNING

        selected_report = safe_report
        selected_variant = "safe_normalized"
    else:
        selected_variant, selected_report, preprocessing_warning = (
            preprocess_with_conservative_trim(
                ffmpeg_path,
                source_path,
                target_path,
                original_report,
                candidate_reports,
            )
        )

    return {
        "selected_output_path": str(target_path),
        "selected_variant": selected_variant,
        "original_duration_seconds": original_duration,
        "selected_duration_seconds": selected_report.get("duration_seconds"),
        "preprocessing_warning": preprocessing_warning,
        "candidate_reports": candidate_reports,
    }


def preprocess_with_conservative_trim(
    ffmpeg_path: str,
    source_path: Path,
    target_path: Path,
    original_report: dict[str, Any],
    candidate_reports: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    """Try a guarded trim candidate, falling back to safe normalized audio."""
    with tempfile.TemporaryDirectory(
        dir=target_path.parent,
        prefix=f".{target_path.stem}-",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        safe_path = temp_dir / "safe_normalized.wav"
        trim_path = temp_dir / "conservative_trim.wav"

        run_ffmpeg_preprocess(
            ffmpeg_path,
            source_path,
            safe_path,
            SAFE_AUDIO_FILTER,
        )
        safe_report = analyze_candidate(safe_path, "Guvenli normalize referans")
        candidate_reports["safe_normalized"] = safe_report

        safe_ok, safe_warning = candidate_is_usable(original_report, safe_report)
        if not safe_ok:
            raise PreprocessingError(safe_warning or "Guvenli on isleme kullanilamadi.")

        selected_path = safe_path
        selected_report = safe_report
        selected_variant = "safe_normalized"
        preprocessing_warning: str | None = None

        run_ffmpeg_preprocess(
            ffmpeg_path,
            source_path,
            trim_path,
            CONSERVATIVE_TRIM_AUDIO_FILTER,
        )
        trim_report = analyze_candidate(trim_path, "Korumaci kirpilmis referans")
        candidate_reports["conservative_trim"] = trim_report

        trim_ok, trim_warning = candidate_is_usable(original_report, trim_report)
        if trim_ok:
            selected_path = trim_path
            selected_report = trim_report
            selected_variant = "conservative_trim"
        else:
            preprocessing_warning = trim_warning or FALLBACK_WARNING

        shutil.copy2(selected_path, target_path)
        final_report = analyze_candidate(target_path, "Secilen on islenmis referans")
        candidate_reports[selected_variant] = final_report
        selected_report = final_report

    return selected_variant, selected_report, preprocessing_warning
