# -*- coding: utf-8 -*-
"""Deneysel fine-tuned XTTS checkpoint ile ilk inference denemesi."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

try:
    from audio_concat_utils import concatenate_wavs
    from text_chunking_utils import split_text_for_tts, summarize_chunks
except ImportError:
    from scripts.audio_concat_utils import concatenate_wavs
    from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR_NAME = "checkpoints"
TRAINING_OUTPUT_DIR_NAME = "training_output"
METADATA_TRAIN_FILE = "metadata_train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetuned_eval"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "finetuned_eval_report.json"
BASE_OUTPUT_PATH = OUTPUT_DIR / "base_test.wav"
FINETUNED_OUTPUT_PATH = OUTPUT_DIR / "finetuned_test.wav"
DEFAULT_TEXT = (
    "Merhaba, bu VoxForge fine-tuning denemesinden sonra oluşturulan ilk test sesidir."
)
LANGUAGE = "tr"
CHUNK_MAX_CHARS = 220
REQUIRED_BASE_FILES = (
    "config.json",
    "vocab.json",
    "model.pth",
    "dvae.pth",
    "mel_stats.pth",
)


class EvaluationError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def resolve_project_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def import_symbol(module_name: str, symbol_name: str) -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return None, f"{symbol_name}: {module_name}: {type(exc).__name__}: {exc}"

    try:
        return getattr(module, symbol_name), None
    except AttributeError:
        return None, f"{symbol_name}: {module_name} icinde bulunamadi"


def import_xtts_api() -> tuple[Any, Any]:
    errors: list[str] = []
    config_class, config_error = import_symbol(
        "TTS.tts.configs.xtts_config",
        "XttsConfig",
    )
    if config_error:
        errors.append(config_error)

    model_class, model_error = import_symbol("TTS.tts.models.xtts", "Xtts")
    if model_error:
        errors.append(model_error)

    if errors:
        raise EvaluationError(
            "XTTS inference importlari basarisiz.\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    print("Import OK: XttsConfig")
    print("Import OK: Xtts")
    return config_class, model_class


def find_selected_checkpoint(experiment_path: Path) -> Path:
    training_output = experiment_path / TRAINING_OUTPUT_DIR_NAME
    if not training_output.is_dir():
        raise EvaluationError(f"training_output klasoru bulunamadi: {training_output}")

    best_models = sorted(
        (path for path in training_output.rglob("best_model.pth") if path.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if best_models:
        return best_models[0]

    numbered_best_models = [
        path
        for path in training_output.rglob("best_model_*.pth")
        if path.is_file()
    ]
    if numbered_best_models:
        return sorted(
            numbered_best_models,
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[0]

    numbered_checkpoints: list[tuple[int, Path]] = []
    for path in training_output.rglob("checkpoint_*.pth"):
        if not path.is_file():
            continue
        match = re.fullmatch(r"checkpoint_(\d+)\.pth", path.name)
        if match:
            numbered_checkpoints.append((int(match.group(1)), path))

    if numbered_checkpoints:
        return sorted(numbered_checkpoints, key=lambda item: item[0], reverse=True)[0][1]

    raise EvaluationError(
        "Fine-tuned checkpoint bulunamadi. Beklenen adaylar: "
        "best_model.pth, best_model_*.pth veya checkpoint_*.pth"
    )


def resolve_base_files(experiment_path: Path) -> dict[str, Path]:
    checkpoint_dir = experiment_path / CHECKPOINT_DIR_NAME
    paths = {name: checkpoint_dir / name for name in REQUIRED_BASE_FILES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise EvaluationError(
            "Experiment checkpoints klasorunde gerekli XTTS dosyalari eksik:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    print("Base XTTS dosyalari OK:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return paths


def read_ljspeech_audio_ids(metadata_path: Path) -> list[str]:
    audio_ids: list[str] = []
    if not metadata_path.is_file():
        return audio_ids

    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        audio_id = stripped.split("|", 1)[0].strip()
        if audio_id:
            audio_ids.append(audio_id)
    return audio_ids


def wav_duration_seconds(wav_path: Path) -> float | None:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return None
            return wav_file.getnframes() / float(sample_rate)
    except (OSError, wave.Error):
        return None


def pick_dataset_speaker_wav(experiment_path: Path) -> Path | None:
    dataset_path = experiment_path / "dataset"
    wavs_path = dataset_path / "wavs"
    metadata_path = dataset_path / METADATA_TRAIN_FILE
    candidates: list[tuple[float, Path]] = []

    for audio_id in read_ljspeech_audio_ids(metadata_path):
        wav_path = wavs_path / f"{audio_id}.wav"
        if not wav_path.is_file():
            continue
        duration = wav_duration_seconds(wav_path)
        if duration is None:
            continue
        candidates.append((duration, wav_path))

    if candidates:
        preferred = [
            (abs(duration - 5.0), duration, wav_path)
            for duration, wav_path in candidates
            if 2.5 <= duration <= 15.0
        ]
        if preferred:
            _score, _duration, wav_path = sorted(preferred, key=lambda item: item[0])[0]
            return wav_path

        return sorted(candidates, key=lambda item: item[0])[0][1]

    fallback_wavs = sorted(wavs_path.glob("*.wav"))
    if fallback_wavs:
        return fallback_wavs[0]
    return None


def resolve_speaker_wav(experiment_path: Path, speaker_wav_arg: str | None) -> Path:
    candidates: list[tuple[str, Path | None]] = []
    if speaker_wav_arg:
        candidates.append(("CLI speaker_wav", resolve_project_path(speaker_wav_arg)))
    else:
        candidates.extend(
            [
                (
                    "profiles/baglare/preprocessed_reference.wav",
                    PROJECT_ROOT / "profiles" / "baglare" / "preprocessed_reference.wav",
                ),
                ("samples/my_voice.wav", PROJECT_ROOT / "samples" / "my_voice.wav"),
                ("dataset kisa wav", pick_dataset_speaker_wav(experiment_path)),
            ]
        )

    for label, path in candidates:
        if path is not None and path.is_file():
            print(f"Speaker wav secildi ({label}): {path}")
            return path.resolve()

    raise EvaluationError(
        "Speaker wav bulunamadi. --speaker-wav verin veya su yollardan birini olusturun: "
        "profiles/baglare/preprocessed_reference.wav, samples/my_voice.wav, "
        "experiments/<run_slug>/dataset/wavs/*.wav"
    )


def load_config(config_class: Any, config_path: Path) -> Any:
    config = config_class()
    if not hasattr(config, "load_json"):
        raise EvaluationError("XttsConfig load_json metodu bulunamadi.")
    config.load_json(str(config_path))
    return config


def set_if_exists(target: Any, field_name: str, value: Any) -> bool:
    if not hasattr(target, field_name):
        return False
    setattr(target, field_name, value)
    return True


def patch_config_paths(config: Any, base_files: dict[str, Path], checkpoint_path: Path) -> None:
    model_args = getattr(config, "model_args", None)
    if model_args is None:
        print("UYARI: config.model_args bulunamadi; path patch atlandi.")
        return

    field_values = {
        "dvae_checkpoint": str(base_files["dvae.pth"]),
        "mel_norm_file": str(base_files["mel_stats.pth"]),
        "xtts_checkpoint": str(base_files["model.pth"]),
        "tokenizer_file": str(base_files["vocab.json"]),
        "gpt_checkpoint": str(checkpoint_path),
        "gpt_checkpoint_path": str(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "vocab_path": str(base_files["vocab.json"]),
    }
    for field_name, value in field_values.items():
        if set_if_exists(model_args, field_name, value):
            print(f"Config model_args path ayarlandi: {field_name}")


def init_model(model_class: Any, config: Any) -> Any:
    if hasattr(model_class, "init_from_config"):
        return model_class.init_from_config(config)
    return model_class(config)


def signature_text(callable_object: Any) -> str:
    try:
        return str(inspect.signature(callable_object))
    except (TypeError, ValueError):
        return "signature okunamadi"


def load_model_checkpoint(
    model: Any,
    config: Any,
    base_files: dict[str, Path],
    checkpoint_path: Path,
    context: str,
) -> None:
    if not hasattr(model, "load_checkpoint"):
        raise EvaluationError(f"{context}: model.load_checkpoint metodu bulunamadi.")

    load_checkpoint = model.load_checkpoint
    print(f"{context} load_checkpoint signature: {signature_text(load_checkpoint)}")
    checkpoint_dir = base_files["config.json"].parent
    attempts = [
        (
            "config + checkpoint_dir + checkpoint_path + vocab_path",
            lambda: load_checkpoint(
                config,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(base_files["vocab.json"]),
                use_deepspeed=False,
            ),
        ),
        (
            "config + checkpoint_path + vocab_path",
            lambda: load_checkpoint(
                config,
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(base_files["vocab.json"]),
                use_deepspeed=False,
            ),
        ),
        (
            "config + checkpoint_dir",
            lambda: load_checkpoint(
                config,
                checkpoint_dir=str(checkpoint_dir),
                use_deepspeed=False,
            ),
        ),
        (
            "keyword config + checkpoint_path + vocab_path",
            lambda: load_checkpoint(
                config=config,
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(base_files["vocab.json"]),
                use_deepspeed=False,
            ),
        ),
        (
            "config + checkpoint_dir + checkpoint_path + vocab_path without use_deepspeed",
            lambda: load_checkpoint(
                config,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(base_files["vocab.json"]),
            ),
        ),
        (
            "config + checkpoint_path + vocab_path without use_deepspeed",
            lambda: load_checkpoint(
                config,
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(base_files["vocab.json"]),
            ),
        ),
        (
            "config + checkpoint_dir without use_deepspeed",
            lambda: load_checkpoint(
                config,
                checkpoint_dir=str(checkpoint_dir),
            ),
        ),
    ]

    errors: list[str] = []
    for label, attempt in attempts:
        try:
            attempt()
            print(f"{context} checkpoint yukleme OK: {label}")
            return
        except TypeError as exc:
            errors.append(f"{label}: TypeError: {exc}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise EvaluationError(
        f"{context}: checkpoint yuklenemedi. Denenen load_checkpoint varyantlari:\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def move_model_to_device(model: Any) -> str:
    try:
        import torch
    except ImportError:
        print("UYARI: torch import edilemedi; model cihaz secimi atlandi.")
        return "unknown"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and hasattr(model, "cuda"):
        model.cuda()
    elif hasattr(model, "to"):
        model.to(device)

    if hasattr(model, "eval"):
        model.eval()

    print(f"Kullanilacak cihaz: {device}")
    return device


def synthesize(model: Any, config: Any, text: str, speaker_wav: Path) -> Any:
    if not hasattr(model, "synthesize"):
        raise EvaluationError("model.synthesize metodu bulunamadi.")

    synthesize_method = model.synthesize
    print(f"synthesize signature: {signature_text(synthesize_method)}")
    attempts = [
        (
            "text, config, speaker_wav, language",
            lambda: synthesize_method(
                text,
                config,
                speaker_wav=str(speaker_wav),
                language=LANGUAGE,
            ),
        ),
        (
            "keyword text/config/speaker_wav/language",
            lambda: synthesize_method(
                text=text,
                config=config,
                speaker_wav=str(speaker_wav),
                language=LANGUAGE,
            ),
        ),
        (
            "speaker_wav list",
            lambda: synthesize_method(
                text,
                config,
                speaker_wav=[str(speaker_wav)],
                language=LANGUAGE,
            ),
        ),
    ]

    errors: list[str] = []
    for label, attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(f"{label}: TypeError: {exc}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise EvaluationError(
        "XTTS synthesize cagrisi basarisiz. Denenen varyantlar:\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def output_sample_rate(config: Any, synth_output: Any) -> int:
    if isinstance(synth_output, dict):
        for key in ("sample_rate", "sampling_rate"):
            value = synth_output.get(key)
            if isinstance(value, int) and value > 0:
                return value

    audio_config = getattr(config, "audio", None)
    for target in (audio_config, config):
        if target is None:
            continue
        for field_name in ("output_sample_rate", "sample_rate"):
            value = getattr(target, field_name, None)
            if isinstance(value, int) and value > 0:
                return value
    return 24000


def flatten_audio_values(audio: Any) -> list[float]:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    if hasattr(audio, "flatten"):
        audio = audio.flatten()
    if hasattr(audio, "tolist"):
        audio = audio.tolist()

    values: list[float] = []

    def collect(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
            return
        values.append(float(item))

    collect(audio)
    return values


def save_wav_from_synthesis(synth_output: Any, output_path: Path, sample_rate: int) -> None:
    if isinstance(synth_output, dict):
        if "wav" in synth_output:
            audio = synth_output["wav"]
        elif "audio" in synth_output:
            audio = synth_output["audio"]
        else:
            raise EvaluationError(
                "synthesize ciktisi icinde 'wav' veya 'audio' alani bulunamadi."
            )
    else:
        audio = synth_output

    values = flatten_audio_values(audio)
    if not values:
        raise EvaluationError("synthesize bos audio verisi dondurdu.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for value in values:
            clipped = max(-1.0, min(1.0, value))
            frames += int(clipped * 32767.0).to_bytes(2, byteorder="little", signed=True)
        wav_file.writeframes(bytes(frames))


def chunk_output_dir(output_path: Path) -> Path:
    return output_path.parent / "chunks" / output_path.stem


def synthesize_to_wav_file(
    model: Any,
    config: Any,
    text: str,
    speaker_wav: Path,
    output_path: Path,
) -> None:
    synth_output = synthesize(model, config, text=text, speaker_wav=speaker_wav)
    sample_rate = output_sample_rate(config, synth_output)
    save_wav_from_synthesis(synth_output, output_path, sample_rate=sample_rate)


def synthesize_text_with_chunking(
    label: str,
    model: Any,
    config: Any,
    text: str,
    speaker_wav: Path,
    output_path: Path,
) -> dict[str, Any]:
    chunks = split_text_for_tts(text, max_chars=CHUNK_MAX_CHARS)
    chunk_summary = summarize_chunks(chunks)

    if not chunks:
        raise EvaluationError("TTS icin bos metin uretildi.")

    if len(chunks) == 1:
        synthesize_to_wav_file(model, config, chunks[0], speaker_wav, output_path)
        return chunk_summary

    chunk_dir = chunk_output_dir(output_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    print(f"{label} chunking kullaniliyor: {len(chunks)} parca")

    for index, chunk in enumerate(chunks, start=1):
        chunk_path = chunk_dir / f"chunk_{index:02d}.wav"
        print(f"{label} chunk_{index:02d}: {len(chunk)} karakter")
        synthesize_to_wav_file(model, config, chunk, speaker_wav, chunk_path)
        chunk_paths.append(chunk_path)

    ok, message = concatenate_wavs(chunk_paths, output_path)
    if not ok:
        raise EvaluationError(message)
    print(f"{label} chunk birlestirme OK: {message}")
    return chunk_summary


def run_xtts_generation(
    label: str,
    config_class: Any,
    model_class: Any,
    base_files: dict[str, Path],
    checkpoint_path: Path,
    speaker_wav: Path,
    text: str,
    output_path: Path,
) -> dict[str, Any]:
    print(f"{label} inference baslatiliyor.")
    print(f"{label} checkpoint: {checkpoint_path}")
    config = load_config(config_class, base_files["config.json"])
    patch_config_paths(config, base_files, checkpoint_path)
    model = init_model(model_class, config)
    load_model_checkpoint(model, config, base_files, checkpoint_path, label)
    move_model_to_device(model)
    chunk_summary = synthesize_text_with_chunking(
        label=label,
        model=model,
        config=config,
        text=text,
        speaker_wav=speaker_wav,
        output_path=output_path,
    )
    print(f"{label} output yazildi: {output_path}")
    return chunk_summary


def initial_report(
    experiment_path: Path,
    selected_checkpoint: Path | None,
    speaker_wav: Path | None,
    text: str,
) -> dict[str, Any]:
    chunk_summary = summarize_chunks(split_text_for_tts(text, max_chars=CHUNK_MAX_CHARS))
    return {
        "experiment": str(experiment_path),
        "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint else None,
        "speaker_wav": str(speaker_wav) if speaker_wav else None,
        "text": text,
        "chunking_used": chunk_summary["chunking_used"],
        "chunk_count": chunk_summary["chunk_count"],
        "chunks": chunk_summary["chunks"],
        "base_output_path": str(BASE_OUTPUT_PATH),
        "finetuned_output_path": str(FINETUNED_OUTPUT_PATH),
        "success": False,
        "errors": [],
        "notes": [
            "Deneysel inference testidir; kalite garantisi vermez.",
            "best_model.pth kalite garantisi degildir.",
            "Base ve fine-tuned ciktilar dinlenerek karsilastirilmalidir.",
            "Kucuk dataset nedeniyle ses benzerligi sinirli olabilir.",
        ],
    }


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Rapor yazildi: {REPORT_PATH}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuned XTTS checkpoint ile deneysel inference testi yapar."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment klasoru. Ornek: experiments/baglare-xtts-exp01",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Turkce test metni.",
    )
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Opsiyonel referans ses yolu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    experiment_path = resolve_project_path(args.experiment)
    text = args.text.strip() or DEFAULT_TEXT
    selected_checkpoint: Path | None = None
    speaker_wav: Path | None = None
    report = initial_report(experiment_path, selected_checkpoint, speaker_wav, text)

    try:
        print("VoxForge fine-tuned XTTS inference testi")
        print(f"Experiment: {experiment_path}")
        print(f"Text: {text}")
        selected_checkpoint = find_selected_checkpoint(experiment_path).resolve()
        report["selected_checkpoint"] = str(selected_checkpoint)
        print(f"Secilen fine-tuned checkpoint: {selected_checkpoint}")

        base_files = resolve_base_files(experiment_path)
        speaker_wav = resolve_speaker_wav(experiment_path, args.speaker_wav)
        report["speaker_wav"] = str(speaker_wav)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config_class, model_class = import_xtts_api()

        try:
            base_chunk_summary = run_xtts_generation(
                "base",
                config_class,
                model_class,
                base_files,
                base_files["model.pth"],
                speaker_wav,
                text,
                BASE_OUTPUT_PATH,
            )
            report["base_chunking"] = base_chunk_summary
            report["notes"].append("Base XTTS output olusturuldu.")
        except Exception as exc:
            error = f"Base XTTS output basarisiz: {type(exc).__name__}: {exc}"
            print(f"UYARI: {error}", file=sys.stderr)
            report["errors"].append(error)

        finetuned_chunk_summary = run_xtts_generation(
            "fine-tuned",
            config_class,
            model_class,
            base_files,
            selected_checkpoint,
            speaker_wav,
            text,
            FINETUNED_OUTPUT_PATH,
        )
        report["finetuned_chunking"] = finetuned_chunk_summary
        report["success"] = True
        report["notes"].append("Fine-tuned checkpoint output olusturuldu.")
        write_report(report)
        print("Fine-tuned checkpoint inference testi tamamlandi.")
        return 0
    except EvaluationError as exc:
        report["errors"].append(str(exc))
        write_report(report)
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["errors"].append(error)
        write_report(report)
        print(f"HATA: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
