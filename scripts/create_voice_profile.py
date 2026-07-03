# -*- coding: utf-8 -*-
"""VoxForge icin yerel voice profile klasoru olusturur."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"

ORIGINAL_REFERENCE_NAME = "original_reference.wav"
PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"
PROFILE_JSON_NAME = "profile.json"

# Gradio demosundaki on isleme ile ayni hedef: mono, 24000 Hz, dengeli WAV.
AUDIO_FILTER = (
    "silenceremove="
    "start_periods=1:start_duration=0.20:start_threshold=-45dB:"
    "stop_periods=1:stop_duration=0.20:stop_threshold=-45dB,"
    "loudnorm=I=-20:TP=-2:LRA=11"
)

TURKISH_CHAR_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "I": "i",
        "İ": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_quality_utils import analyze_audio_file


class VoiceProfileError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def slugify_profile_name(profile_name: str) -> str:
    """Profil adini guvenli klasor adina cevirir."""
    normalized = profile_name.strip().translate(TURKISH_CHAR_MAP).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    if not normalized:
        raise VoiceProfileError(
            "Profil adi guvenli klasor adina cevrilemedi. "
            "Lutfen harf veya rakam iceren bir ad kullanin."
        )

    return normalized


def resolve_input_path(input_path: str) -> Path:
    """Giris dosyasini proje kokune gore cozer."""
    candidate = Path(input_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    return candidate.resolve()


def ensure_profile_path_is_safe(profile_dir: Path) -> None:
    """Profil yolunun profiles klasoru disina cikmadigini dogrular."""
    profiles_root = PROFILES_DIR.resolve()
    resolved_profile_dir = profile_dir.resolve()

    try:
        resolved_profile_dir.relative_to(profiles_root)
    except ValueError as exc:
        raise VoiceProfileError("Profil yolu profiles klasoru disina cikamaz.") from exc


def build_preprocess_command(
    ffmpeg_path: str,
    source_audio: Path,
    output_audio: Path,
) -> list[str]:
    """Referans sesi XTTS icin standart WAV formata hazirlayan komutu kurar."""
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(source_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        AUDIO_FILTER,
        "-c:a",
        "pcm_s16le",
        str(output_audio),
    ]


def run_preprocess(
    ffmpeg_path: str,
    source_audio: Path,
    output_audio: Path,
) -> None:
    """FFmpeg ile on islenmis referans WAV dosyasini uretir."""
    command = build_preprocess_command(ffmpeg_path, source_audio, output_audio)
    print("FFmpeg on isleme komutu:")
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
        raise VoiceProfileError(
            "FFmpeg referans ses on isleme adimi basarisiz oldu.\n"
            f"Ayrinti: {detail}"
        )

    if not output_audio.is_file():
        raise VoiceProfileError(
            "FFmpeg tamamlandi ancak preprocessed_reference.wav olusmadi."
        )


def analyze_or_fail(audio_path: Path, label: str) -> dict[str, Any]:
    """Kalite analizini calistirir ve beklenmeyen hatalari sade mesajla sarar."""
    try:
        return analyze_audio_file(audio_path)
    except Exception as exc:
        raise VoiceProfileError(f"{label} kalite analizi basarisiz oldu: {exc}") from exc


def project_relative_path(path: Path) -> str:
    """Profile JSON icin proje kokune gore tasinabilir yol dondurur."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_profile_payload(
    profile_name: str,
    profile_slug: str,
    original_reference: Path,
    preprocessed_reference: Path,
    original_quality: dict[str, Any],
    preprocessed_quality: dict[str, Any],
) -> dict[str, Any]:
    """profile.json icin yazilacak metadata verisini hazirlar."""
    return {
        "profile_name": profile_name,
        "profile_slug": profile_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_reference_path": project_relative_path(original_reference),
        "preprocessed_reference_path": project_relative_path(preprocessed_reference),
        "original_quality": original_quality,
        "preprocessed_quality": preprocessed_quality,
        "notes": [
            "Bu profil yerel kullanim icindir; profiles/* GitHub'a yuklenmez.",
            "original_reference.wav giris sesinin kopyasidir.",
            "preprocessed_reference.wav FFmpeg ile mono 24000 Hz pcm_s16le WAV olarak olusturulur.",
        ],
    }


def create_voice_profile(profile_name: str, input_path: str) -> Path:
    """Profil klasorunu, referans sesleri ve profile.json dosyasini olusturur."""
    profile_slug = slugify_profile_name(profile_name)
    source_audio = resolve_input_path(input_path)
    profile_dir = PROFILES_DIR / profile_slug
    ensure_profile_path_is_safe(profile_dir)

    if not source_audio.is_file():
        raise VoiceProfileError(f"Giris ses dosyasi bulunamadi: {source_audio}")

    if profile_dir.exists():
        raise VoiceProfileError(
            "Bu isimle bir voice profile zaten var; uzerine yazilmadi.\n"
            f"Mevcut profil klasoru: {profile_dir}"
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise VoiceProfileError(
            "FFmpeg bulunamadi. Gyan.FFmpeg.Shared kurulumunu ve PATH ayarini kontrol edin."
        )

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    temp_profile_dir = Path(
        tempfile.mkdtemp(dir=PROFILES_DIR, prefix=f".{profile_slug}-")
    )
    profile_created = False

    try:
        original_reference = temp_profile_dir / ORIGINAL_REFERENCE_NAME
        preprocessed_reference = temp_profile_dir / PREPROCESSED_REFERENCE_NAME
        profile_json = temp_profile_dir / PROFILE_JSON_NAME

        print(f"Profil adi: {profile_name}")
        print(f"Profil slug: {profile_slug}")
        print(f"Giris ses dosyasi: {source_audio}")
        print(f"Gecici profil klasoru: {temp_profile_dir}")

        shutil.copy2(source_audio, original_reference)
        run_preprocess(ffmpeg_path, original_reference, preprocessed_reference)

        original_quality = analyze_or_fail(original_reference, "Orijinal referans")
        preprocessed_quality = analyze_or_fail(
            preprocessed_reference,
            "On islenmis referans",
        )

        payload = build_profile_payload(
            profile_name=profile_name,
            profile_slug=profile_slug,
            original_reference=profile_dir / ORIGINAL_REFERENCE_NAME,
            preprocessed_reference=profile_dir / PREPROCESSED_REFERENCE_NAME,
            original_quality=original_quality,
            preprocessed_quality=preprocessed_quality,
        )
        profile_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temp_profile_dir.rename(profile_dir)
        profile_created = True
    finally:
        if not profile_created and temp_profile_dir.exists():
            shutil.rmtree(temp_profile_dir, ignore_errors=True)

    print("Voice profile olusturuldu.")
    print(f"Profil klasoru: {profile_dir}")
    print(f"Orijinal kalite: {original_quality.get('quality', 'UNKNOWN')}")
    print(f"On islenmis kalite: {preprocessed_quality.get('quality', 'UNKNOWN')}")
    print(f"Metadata: {profile_dir / PROFILE_JSON_NAME}")
    return profile_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge icin yerel voice profile olusturur."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Profil adi. Guvenli klasor adina cevrilir.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Kullanilacak referans ses dosyasi yolu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        create_voice_profile(args.name, args.input)
    except VoiceProfileError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
