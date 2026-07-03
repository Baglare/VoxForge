# -*- coding: utf-8 -*-
"""XTTS icin referans sesi standart wav formatina hazirlar."""

from pathlib import Path
import shutil
import subprocess
import sys
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
OUTPUT_AUDIO = PROJECT_ROOT / "outputs" / "preprocessed_reference.wav"

# Sessizlik kirpma esikleri dusuk tutuldu; amac sadece bastaki/sondaki bosluklari azaltmak.
# loudnorm hedefleri de bilincli olarak orta seviyede, agresif yukseklik artisi yapmaz.
AUDIO_FILTER = (
    "silenceremove="
    "start_periods=1:start_duration=0.20:start_threshold=-45dB:"
    "stop_periods=1:stop_duration=0.20:stop_threshold=-45dB,"
    "loudnorm=I=-20:TP=-2:LRA=11"
)


def format_file_size(byte_count: int) -> str:
    """Dosya boyutunu okunabilir bicimde dondurur."""
    if byte_count < 1024:
        return f"{byte_count} B"

    kibibytes = byte_count / 1024
    if kibibytes < 1024:
        return f"{kibibytes:.1f} KB"

    mebibytes = kibibytes / 1024
    return f"{mebibytes:.2f} MB"


def get_wav_duration_seconds(audio_path: Path) -> float | None:
    """WAV dosyasinin tahmini suresini saniye olarak hesaplar."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
    except (EOFError, OSError, wave.Error):
        return None

    if frame_rate <= 0:
        return None

    return frame_count / float(frame_rate)


def build_ffmpeg_command(ffmpeg_path: str) -> list[str]:
    """FFmpeg komutunu arguman listesi olarak hazirlar."""
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(INPUT_AUDIO),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        AUDIO_FILTER,
        "-c:a",
        "pcm_s16le",
        str(OUTPUT_AUDIO),
    ]


def main() -> int:
    print("VoxForge referans ses hazirlama scripti baslatiliyor.")
    print(f"Giris dosyasi: {INPUT_AUDIO}")
    print(f"Cikis dosyasi: {OUTPUT_AUDIO}")

    input_exists = INPUT_AUDIO.is_file()
    print(f"Giris dosyasi var mi: {'evet' if input_exists else 'hayir'}")

    if not input_exists:
        print(
            "HATA: Referans ses dosyasi bulunamadi. "
            f"Lutfen dosyayi buraya ekleyin: {INPUT_AUDIO}",
            file=sys.stderr,
        )
        return 1

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print(
            "HATA: FFmpeg bulunamadi. Lutfen FFmpeg'i kurup PATH icine ekleyin.",
            file=sys.stderr,
        )
        return 1

    # Cikti klasoru yoksa olusturulur.
    OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)

    command = build_ffmpeg_command(ffmpeg_path)
    print("Kullanilan FFmpeg komutu:")
    print(subprocess.list2cmdline(command))

    print("Ses hazirlama baslatiliyor...")
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        print("HATA: FFmpeg ses isleme adimi basarisiz oldu.", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        print("Cikis dosyasi olusturuldu mu: hayir")
        return result.returncode

    output_created = OUTPUT_AUDIO.is_file()
    print(f"Cikis dosyasi olusturuldu mu: {'evet' if output_created else 'hayir'}")

    if not output_created:
        print("HATA: FFmpeg tamamlandi ancak cikis dosyasi bulunamadi.", file=sys.stderr)
        return 1

    duration_seconds = get_wav_duration_seconds(OUTPUT_AUDIO)
    if duration_seconds is None:
        print("Tahmini sure: hesaplanamadi")
    else:
        print(f"Tahmini sure: {duration_seconds:.2f} saniye")

    output_size = OUTPUT_AUDIO.stat().st_size
    print(f"Dosya boyutu: {format_file_size(output_size)} ({output_size} bayt)")
    print("Referans ses hazirlama tamamlandi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
