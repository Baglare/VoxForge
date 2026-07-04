# -*- coding: utf-8 -*-
"""Deneysel XTTS-v2 GPT encoder fine-tuning baslatma scripti."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = "experiment_manifest.json"
CHECKPOINT_DIR_NAME = "checkpoints"
TRAINING_OUTPUT_DIR_NAME = "training_output"
METADATA_TRAIN_FILE = "metadata_train.csv"
METADATA_EVAL_FILE = "metadata_eval.csv"
REQUIRED_XTTS_FILES = (
    "model.pth",
    "config.json",
    "vocab.json",
    "dvae.pth",
    "mel_stats.pth",
)

# Coqui XTTS-v2 ana dosyalari. Script bunlari kullanici training komutunu
# calistirdiginda experiments/<run_slug>/checkpoints altina indirmeyi dener.
XTTS_FILE_URLS = {
    "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
    "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
}


class TrainingError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def resolve_experiment_path(experiment_arg: str) -> Path:
    """Experiment yolunu proje kokune gore cozer."""
    experiment_path = Path(experiment_arg)
    if not experiment_path.is_absolute():
        experiment_path = PROJECT_ROOT / experiment_path
    return experiment_path.resolve()


def read_manifest(experiment_path: Path) -> dict[str, Any]:
    """Experiment manifest dosyasini okur."""
    manifest_path = experiment_path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise TrainingError(f"experiment_manifest.json bulunamadi: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"experiment_manifest.json okunamadi: {exc}") from exc

    required_keys = ("run_slug", "train_samples", "language")
    missing_keys = [key for key in required_keys if key not in manifest]
    if missing_keys:
        raise TrainingError(
            "experiment_manifest.json icinde kritik alan eksik: "
            + ", ".join(missing_keys)
        )

    return manifest


def get_cuda_info() -> tuple[bool, str]:
    """Torch varsa CUDA durumunu okunabilir sekilde dondurur."""
    try:
        import torch
    except ImportError:
        return False, "torch import edilemedi"

    available = bool(torch.cuda.is_available())
    if not available:
        return False, "CUDA kullanilabilir degil"

    try:
        return True, torch.cuda.get_device_name(0)
    except Exception as exc:
        return True, f"CUDA cihaz adi okunamadi: {type(exc).__name__}: {exc}"


def print_preflight(
    experiment_path: Path,
    manifest: dict[str, Any],
    max_steps: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Training baslamadan once kritik bilgileri terminale basar."""
    dataset_path = experiment_path / "dataset"
    cuda_available, cuda_device_name = get_cuda_info()

    print("Deneysel XTTS GPT fine-tuning")
    print(f"Experiment path: {experiment_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Train sample count: {manifest.get('train_samples')}")
    print(f"Eval sample count: {manifest.get('eval_samples')}")
    print(f"Language: {manifest.get('language')}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accumulation: {grad_accum}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device name: {cuda_device_name}")
    print("")


def import_training_api() -> dict[str, Any]:
    """Coqui/Trainer importlarini toplar ve eksikleri acik sekilde raporlar."""
    missing: list[str] = []
    api: dict[str, Any] = {}

    try:
        from trainer import Trainer, TrainerArgs
    except ImportError as exc:
        missing.append(f"Trainer / TrainerArgs: {exc}")
    else:
        api["Trainer"] = Trainer
        api["TrainerArgs"] = TrainerArgs
        print("Import OK: Trainer, TrainerArgs")

    try:
        from TTS.config.shared_configs import BaseDatasetConfig
    except ImportError:
        try:
            from TTS.tts.configs.shared_configs import BaseDatasetConfig
        except ImportError as exc:
            missing.append(f"BaseDatasetConfig: {exc}")
        else:
            api["BaseDatasetConfig"] = BaseDatasetConfig
            print("Import OK: BaseDatasetConfig")
    else:
        api["BaseDatasetConfig"] = BaseDatasetConfig
        print("Import OK: BaseDatasetConfig")

    try:
        from TTS.tts.datasets import load_tts_samples
    except ImportError as exc:
        missing.append(f"load_tts_samples: {exc}")
    else:
        api["load_tts_samples"] = load_tts_samples
        print("Import OK: load_tts_samples")

    for symbol_name in ("GPTArgs", "GPTTrainer", "GPTTrainerConfig"):
        imported_symbol, error = import_symbol(
            "TTS.tts.layers.xtts.trainer.gpt_trainer",
            symbol_name,
        )
        if error:
            missing.append(error)
            continue

        api[symbol_name] = imported_symbol
        print(f"Import OK: {symbol_name}")

    audio_config_class, audio_config_source, audio_config_errors = import_xtts_audio_config()
    if audio_config_class is None:
        missing.append(
            "XttsAudioConfig import edilemedi. Denenen moduller:\n"
            + "\n".join(f"  - {error}" for error in audio_config_errors)
        )
    else:
        api["XttsAudioConfig"] = audio_config_class
        api["XttsAudioConfigSource"] = audio_config_source
        print("Import OK: XttsAudioConfig")
        print(f"XttsAudioConfig import source: {audio_config_source}")

    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise TrainingError(
            "Coqui TTS / trainer importlari eksik veya mevcut paket API'si farkli.\n"
            f"{detail}\n"
            "Kurulu coqui-tts surumunu ve official XTTS GPT training recipe'sini kontrol edin."
        )

    return api


def import_symbol(module_name: str, symbol_name: str) -> tuple[Any | None, str | None]:
    """Tek bir sembolu import eder; AttributeError zinciri diger importlari bozmaz."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return None, f"{symbol_name}: {module_name}: {exc}"

    try:
        return getattr(module, symbol_name), None
    except AttributeError:
        return None, f"{symbol_name}: {module_name} icinde bulunamadi"


def import_xtts_audio_config() -> tuple[Any | None, str | None, list[str]]:
    """XttsAudioConfig sinifini Coqui surumleri arasinda fallback ile bulur."""
    candidate_modules = (
        "TTS.tts.layers.xtts.trainer.gpt_trainer",
        "TTS.tts.models.xtts",
        "TTS.tts.configs.xtts_config",
    )
    errors: list[str] = []

    for module_name in candidate_modules:
        imported_symbol, error = import_symbol(module_name, "XttsAudioConfig")
        if error:
            errors.append(error)
            continue
        return imported_symbol, module_name, errors

    return None, None, errors


def get_signature(callable_object: Any) -> inspect.Signature | None:
    """Signature okunamiyorsa None dondurur."""
    try:
        return inspect.signature(callable_object)
    except (TypeError, ValueError):
        return None


def filter_supported_kwargs(
    callable_object: Any,
    kwargs: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    """Kurulu API'nin desteklemedigi kwargs degerlerini terminale yazip atlar."""
    signature = get_signature(callable_object)
    if signature is None:
        print(f"UYARI: {context} signature okunamadi; tum kwargs deneniyor.")
        return dict(kwargs)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs)

    supported: dict[str, Any] = {}
    unsupported: list[str] = []
    for key, value in kwargs.items():
        if key in signature.parameters:
            supported[key] = value
        else:
            unsupported.append(key)

    for key in unsupported:
        print(f"UYARI: {context} desteklemeyen arguman atlandi: {key}")

    missing_required = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is inspect.Parameter.empty and name not in supported:
            missing_required.append(name)

    if missing_required:
        raise TrainingError(
            f"{context} icin kritik zorunlu arguman eksik: "
            + ", ".join(missing_required)
        )

    return supported


def unexpected_keyword_from_type_error(exc: TypeError) -> str | None:
    """TypeError icinden beklenmeyen keyword adini yakalar."""
    match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
    if match:
        return match.group(1)
    return None


def instantiate_supported(
    callable_object: Any,
    kwargs: dict[str, Any],
    context: str,
) -> Any:
    """Signature kontroluyle nesne olusturur ve API farklarini acik raporlar."""
    supported_kwargs = filter_supported_kwargs(callable_object, kwargs, context)
    return instantiate_filtered(callable_object, supported_kwargs, context)


def instantiate_filtered(
    callable_object: Any,
    supported_kwargs: dict[str, Any],
    context: str,
) -> Any:
    """Onceden filtrelenmis kwargs ile nesne olusturur."""
    while True:
        try:
            return callable_object(**supported_kwargs)
        except TypeError as exc:
            unexpected_keyword = unexpected_keyword_from_type_error(exc)
            if unexpected_keyword and unexpected_keyword in supported_kwargs:
                print(
                    f"UYARI: {context} TypeError verdi; "
                    f"desteklenmeyen arguman atlandi: {unexpected_keyword}"
                )
                supported_kwargs.pop(unexpected_keyword)
                continue

            keyword_detail = (
                f"\nPatlayan keyword: {unexpected_keyword}"
                if unexpected_keyword
                else ""
            )
            raise TrainingError(
                f"{context} olusturulamadi.\n"
                f"Hata: TypeError: {exc}{keyword_detail}\n"
                "Kurulu Coqui TTS API'si official XTTS GPT training recipe ile farkli olabilir."
            ) from exc


def download_file(url: str, target_path: Path) -> None:
    """Tek checkpoint dosyasini stdlib ile indirir."""
    print(f"Indiriliyor: {target_path.name}")
    try:
        with urllib.request.urlopen(url) as response:
            with target_path.open("wb") as target_file:
                shutil.copyfileobj(response, target_file)
    except (urllib.error.URLError, OSError) as exc:
        if target_path.exists():
            target_path.unlink()
        raise TrainingError(
            f"Checkpoint indirilemedi: {target_path.name}\n"
            f"URL: {url}\n"
            f"Hata: {type(exc).__name__}: {exc}"
        ) from exc


def checkpoint_paths(checkpoint_dir: Path) -> dict[str, Path]:
    """Gerekli checkpoint dosya yollarini dondurur."""
    return {name: checkpoint_dir / name for name in REQUIRED_XTTS_FILES}


def check_checkpoint_files(checkpoint_dir: Path) -> dict[str, Path]:
    """Dry-run icin checkpoint dosyalarinin mevcut oldugunu kontrol eder."""
    paths = checkpoint_paths(checkpoint_dir)
    missing_files = []

    for name, path in paths.items():
        if path.is_file():
            print(f"Checkpoint OK: {path}")
        else:
            print(f"Checkpoint eksik: {path}")
            missing_files.append(name)

    if missing_files:
        raise TrainingError(
            "Dry-run config kontrolu icin checkpoint dosyalari eksik: "
            + ", ".join(missing_files)
        )

    return paths


def ensure_xtts_files(checkpoint_dir: Path) -> dict[str, Path]:
    """Gerekli XTTS dosyalari yoksa checkpoints klasorune indirir."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(checkpoint_dir)

    for name, url in XTTS_FILE_URLS.items():
        target_path = paths[name]
        if target_path.is_file():
            print(f"Checkpoint mevcut: {target_path}")
            continue
        download_file(url, target_path)

    return paths


def validate_experiment_layout(experiment_path: Path) -> tuple[Path, Path, Path | None]:
    """Dataset klasoru ve metadata dosyalarini kontrol eder."""
    dataset_path = experiment_path / "dataset"
    train_metadata_path = dataset_path / METADATA_TRAIN_FILE
    eval_metadata_path = dataset_path / METADATA_EVAL_FILE
    wavs_path = dataset_path / "wavs"

    if not experiment_path.is_dir():
        raise TrainingError(f"Experiment klasoru bulunamadi: {experiment_path}")
    if not dataset_path.is_dir():
        raise TrainingError(f"Experiment dataset klasoru bulunamadi: {dataset_path}")
    if not wavs_path.is_dir():
        raise TrainingError(f"Experiment WAV klasoru bulunamadi: {wavs_path}")
    if not train_metadata_path.is_file():
        raise TrainingError(f"Train metadata bulunamadi: {train_metadata_path}")

    print(f"Dataset OK: {dataset_path}")
    print(f"Train metadata OK: {train_metadata_path}")

    if eval_metadata_path.is_file():
        print(f"Eval metadata OK: {eval_metadata_path}")
        return dataset_path, train_metadata_path, eval_metadata_path

    print(
        "UYARI: metadata_eval.csv bulunamadi; "
        "train metadata uzerinden eval split kullanilacak."
    )
    return dataset_path, train_metadata_path, None


def make_dataset_config(
    api: dict[str, Any],
    experiment_path: Path,
    manifest: dict[str, Any],
    eval_metadata_path: Path | None,
) -> tuple[Any, bool]:
    """Coqui LJSpeech dataset config nesnesini olusturur."""
    dataset_path = experiment_path / "dataset"
    kwargs = {
        "formatter": "ljspeech",
        "dataset_name": manifest["run_slug"],
        "path": str(dataset_path),
        "meta_file_train": METADATA_TRAIN_FILE,
        "language": manifest.get("language", "tr"),
    }
    if eval_metadata_path is not None:
        kwargs["meta_file_val"] = METADATA_EVAL_FILE

    supported_kwargs = filter_supported_kwargs(
        api["BaseDatasetConfig"],
        kwargs,
        "BaseDatasetConfig",
    )
    dataset_config = instantiate_supported(
        api["BaseDatasetConfig"],
        supported_kwargs,
        "BaseDatasetConfig",
    )

    eval_metadata_used = eval_metadata_path is not None and "meta_file_val" in supported_kwargs
    if eval_metadata_path is not None and not eval_metadata_used:
        print(
            "UYARI: BaseDatasetConfig meta_file_val desteklemiyor; "
            "train metadata uzerinden eval split kullanilacak."
        )

    return dataset_config, eval_metadata_used


def read_ljspeech_audio_ids(metadata_path: Path) -> list[str]:
    """Basliksiz LJSpeech metadata dosyasindan audio id listesi okur."""
    audio_ids: list[str] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        audio_id = stripped.split("|", 1)[0].strip()
        if audio_id:
            audio_ids.append(audio_id)
    return audio_ids


def wav_duration_seconds(wav_path: Path) -> float | None:
    """WAV suresini stdlib wave moduluyle okumayi dener."""
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return None
            return wav_file.getnframes() / float(sample_rate)
    except (OSError, wave.Error):
        return None


def pick_speaker_wav(dataset_path: Path, train_metadata_path: Path) -> Path:
    """Test sentences icin dataset icinden kisa ve temiz bir WAV secer."""
    wavs_path = dataset_path / "wavs"
    candidates: list[tuple[float, Path]] = []

    for audio_id in read_ljspeech_audio_ids(train_metadata_path):
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
            _score, duration, wav_path = sorted(preferred, key=lambda item: item[0])[0]
            print(f"Test speaker_wav: {wav_path} ({duration:.2f} sn)")
            return wav_path

        duration, wav_path = sorted(candidates, key=lambda item: item[0])[0]
        print(f"Test speaker_wav: {wav_path} ({duration:.2f} sn)")
        return wav_path

    fallback_wavs = sorted(wavs_path.glob("*.wav"))
    if fallback_wavs:
        print(f"Test speaker_wav: {fallback_wavs[0]}")
        return fallback_wavs[0]

    raise TrainingError(f"Test speaker_wav icin WAV bulunamadi: {wavs_path}")


def make_test_sentences(speaker_wav: Path) -> list[dict[str, str]]:
    """XTTS config icin kisa Turkce test cumleleri olusturur."""
    return [
        {
            "text": "Bugun VoxForge ile deneysel bir fine tuning denemesi yapiyorum.",
            "speaker_wav": str(speaker_wav),
            "language": "tr",
        },
        {
            "text": "Bu cikti yalnizca yerel ve deneysel degerlendirme icindir.",
            "speaker_wav": str(speaker_wav),
            "language": "tr",
        },
    ]


def build_gpt_args(api: dict[str, Any], checkpoint_files: dict[str, Path]) -> Any:
    """GPTArgs nesnesini official GPT trainer recipe mantigiyla olusturur."""
    kwargs = {
        "max_conditioning_length": 132300,
        "min_conditioning_length": 66150,
        "debug_loading_failures": False,
        "max_wav_length": 255995,
        "max_text_length": 200,
        "mel_norm_file": str(checkpoint_files["mel_stats.pth"]),
        "dvae_checkpoint": str(checkpoint_files["dvae.pth"]),
        "xtts_checkpoint": str(checkpoint_files["model.pth"]),
        "tokenizer_file": str(checkpoint_files["vocab.json"]),
        "gpt_num_audio_tokens": 1026,
        "gpt_start_audio_token": 1024,
        "gpt_stop_audio_token": 1025,
        "gpt_use_masking_gt_prompt_approach": True,
        "gpt_use_perceiver_resampler": True,
    }
    return instantiate_supported(api["GPTArgs"], kwargs, "GPTArgs")


def build_audio_config(api: dict[str, Any]) -> Any:
    """XTTS audio config nesnesini olusturur."""
    kwargs = {
        "sample_rate": 22050,
        "dvae_sample_rate": 22050,
        "output_sample_rate": 24000,
    }
    audio_config_class = api["XttsAudioConfig"]
    audio_config_source = api.get("XttsAudioConfigSource", "bilinmiyor")
    supported_kwargs = filter_supported_kwargs(
        audio_config_class,
        kwargs,
        "XttsAudioConfig",
    )
    supported_names = ", ".join(supported_kwargs.keys()) if supported_kwargs else "yok"
    print(f"DEBUG: XttsAudioConfig source: {audio_config_source}")
    print(f"DEBUG: XttsAudioConfig desteklenen alanlar: {supported_names}")
    return instantiate_filtered(audio_config_class, supported_kwargs, "XttsAudioConfig")


def build_gpt_trainer_config(
    api: dict[str, Any],
    manifest: dict[str, Any],
    dataset_config: Any,
    checkpoint_files: dict[str, Path],
    output_path: Path,
    batch_size: int,
    max_steps: int,
    speaker_wav: Path,
) -> Any:
    """GPTTrainerConfig nesnesini kucuk deneysel varsayilanlarla olusturur."""
    model_args = build_gpt_args(api, checkpoint_files)
    audio_config = build_audio_config(api)
    test_sentences = make_test_sentences(speaker_wav)
    save_step = max(25, min(max_steps, 100))

    kwargs = {
        "output_path": str(output_path),
        "model_args": model_args,
        "run_name": f"{manifest['run_slug']}_gpt",
        "project_name": "VoxForge_XTTS_GPT_Experiment",
        "run_description": "Experimental local XTTS-v2 GPT encoder fine-tuning.",
        "dashboard_logger": "tensorboard",
        "logger_uri": None,
        "audio": audio_config,
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "batch_group_size": 16,
        "num_loader_workers": 0,
        "num_eval_loader_workers": 0,
        "eval_split_max_size": 256,
        "print_step": 25,
        "plot_step": 100,
        "log_model_step": 100,
        "save_step": save_step,
        "save_n_checkpoints": 1,
        "save_checkpoints": True,
        "print_eval": False,
        "optimizer": "AdamW",
        "optimizer_wd_only_on_weights": True,
        "optimizer_params": {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        "lr": 5e-6,
        "lr_scheduler": "MultiStepLR",
        "lr_scheduler_params": {"milestones": [max_steps + 1], "gamma": 0.5, "last_epoch": -1},
        "datasets": [dataset_config],
        "test_sentences": test_sentences,
    }

    config = instantiate_supported(api["GPTTrainerConfig"], kwargs, "GPTTrainerConfig")

    # Bazı Coqui surumleri constructor'da desteklemedigi alanlari daha sonra
    # attribute olarak kabul eder. Kritik alanlari burada da set etmeyi deneriz.
    for name, value in {
        "datasets": [dataset_config],
        "test_sentences": test_sentences,
    }.items():
        if not hasattr(config, name):
            print(f"UYARI: GPTTrainerConfig attribute bulunamadi: {name}")
            continue
        setattr(config, name, value)

    if hasattr(config, "max_steps"):
        setattr(config, "max_steps", max_steps)
        print(f"Max steps GPTTrainerConfig uzerinde ayarlandi: {max_steps}")
    else:
        print(
            "UYARI: GPTTrainerConfig max_steps alani desteklemiyor. "
            "TrainerArgs desteklemezse egitim adim sayisi kesin sinirlanmayabilir."
        )

    return config


def build_trainer_args(
    TrainerArgs: Any,
    max_steps: int,
    grad_accum: int,
    start_with_eval: bool,
) -> Any:
    """TrainerArgs nesnesini surum farklarina toleransli sekilde olusturur."""
    kwargs = {
        "restore_path": None,
        "skip_train_epoch": False,
        "start_with_eval": start_with_eval,
        "grad_accum_steps": grad_accum,
        "max_steps": max_steps,
    }
    trainer_args = instantiate_supported(TrainerArgs, kwargs, "TrainerArgs")

    if hasattr(trainer_args, "grad_accum_steps"):
        setattr(trainer_args, "grad_accum_steps", grad_accum)
        print(f"Grad accumulation TrainerArgs uzerinde ayarlandi: {grad_accum}")
    else:
        print("UYARI: TrainerArgs grad_accum_steps alani desteklemiyor.")

    if hasattr(trainer_args, "max_steps"):
        setattr(trainer_args, "max_steps", max_steps)
        print(f"Max steps TrainerArgs uzerinde ayarlandi: {max_steps}")
    else:
        print(
            "UYARI: TrainerArgs max_steps alani desteklemiyor. "
            "Kurulu trainer surumu max_steps limitini dogrudan uygulamayabilir."
        )

    return trainer_args


def load_samples(
    api: dict[str, Any],
    dataset_config: Any,
    manifest: dict[str, Any],
    eval_metadata_used: bool,
) -> tuple[list[Any], list[Any]]:
    """Coqui dataset loader ile train/eval sample listelerini yukler."""
    load_tts_samples = api["load_tts_samples"]
    eval_sample_count = max(1, int(manifest.get("eval_samples", 1)))

    if not eval_metadata_used:
        print(
            "Eval metadata dogrudan kullanilamiyor; "
            "load_tts_samples train metadata uzerinden eval split uretecek."
        )

    kwargs = {
        "eval_split": True,
        "eval_split_max_size": eval_sample_count,
        "eval_split_size": 0.10,
    }

    try:
        samples = load_tts_samples([dataset_config], **kwargs)
    except TypeError:
        samples = load_tts_samples(dataset_config, **kwargs)

    if isinstance(samples, tuple):
        return samples[0], samples[1]
    return samples, []


def create_gpt_model(GPTTrainer: Any, config: Any) -> Any:
    """GPTTrainer model nesnesini official recipe yoluyla olusturur."""
    if hasattr(GPTTrainer, "init_from_config"):
        return GPTTrainer.init_from_config(config)
    return GPTTrainer(config)


def prepare_training_objects(
    experiment_path: Path,
    manifest: dict[str, Any],
    checkpoint_files: dict[str, Path],
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    dry_run: bool,
) -> tuple[dict[str, Any], Any, Any, Any, Path, bool]:
    """Dry-run ve training icin ortak import/config kontrollerini yapar."""
    dataset_path, train_metadata_path, eval_metadata_path = validate_experiment_layout(experiment_path)
    api = import_training_api()
    dataset_config, eval_metadata_used = make_dataset_config(
        api,
        experiment_path,
        manifest,
        eval_metadata_path,
    )
    speaker_wav = pick_speaker_wav(dataset_path, train_metadata_path)
    output_path = experiment_path / TRAINING_OUTPUT_DIR_NAME
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    config = build_gpt_trainer_config(
        api,
        manifest,
        dataset_config,
        checkpoint_files,
        output_path,
        batch_size,
        max_steps,
        speaker_wav,
    )
    trainer_args = build_trainer_args(
        api["TrainerArgs"],
        max_steps=max_steps,
        grad_accum=grad_accum,
        start_with_eval=bool(eval_metadata_path),
    )

    print("GPT trainer config olusturma OK.")
    if dry_run:
        print("Dry-run modu: load_tts_samples ve training baslatilmadi.")

    return api, dataset_config, config, trainer_args, output_path, eval_metadata_used


def run_dry_run(
    experiment_path: Path,
    manifest: dict[str, Any],
    max_steps: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Training baslatmadan dosya/import/config kontrollerini yapar."""
    checkpoint_files = check_checkpoint_files(experiment_path / CHECKPOINT_DIR_NAME)
    prepare_training_objects(
        experiment_path,
        manifest,
        checkpoint_files,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        dry_run=True,
    )
    print("XTTS fine-tuning dry-run completed successfully")


def run_training(
    experiment_path: Path,
    manifest: dict[str, Any],
    max_steps: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Coqui Trainer ile deneysel XTTS GPT training baslatir."""
    checkpoint_files = ensure_xtts_files(experiment_path / CHECKPOINT_DIR_NAME)
    api, dataset_config, config, trainer_args, output_path, eval_metadata_used = prepare_training_objects(
        experiment_path,
        manifest,
        checkpoint_files,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        dry_run=False,
    )

    train_samples, eval_samples = load_samples(api, dataset_config, manifest, eval_metadata_used)
    print(f"Coqui train samples: {len(train_samples)}")
    print(f"Coqui eval samples: {len(eval_samples)}")

    model = create_gpt_model(api["GPTTrainer"], config)
    trainer = api["Trainer"](
        trainer_args,
        config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deneysel XTTS-v2 GPT encoder fine-tuning baslatir."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment klasoru. Ornek: experiments/baglare-xtts-exp01",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Manifest, dosya, import ve config kontrolu yap; training baslatma.",
    )
    return parser.parse_args(argv)


def validate_cli_args(max_steps: int, batch_size: int, grad_accum: int) -> None:
    """Temel CLI sayisal degerlerini kontrol eder."""
    if max_steps <= 0:
        raise TrainingError("--max-steps 0'dan buyuk olmali.")
    if batch_size <= 0:
        raise TrainingError("--batch-size 0'dan buyuk olmali.")
    if grad_accum <= 0:
        raise TrainingError("--grad-accum 0'dan buyuk olmali.")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    experiment_path = resolve_experiment_path(args.experiment)

    try:
        validate_cli_args(args.max_steps, args.batch_size, args.grad_accum)
        manifest = read_manifest(experiment_path)
        print_preflight(
            experiment_path,
            manifest,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        if args.dry_run:
            run_dry_run(
                experiment_path,
                manifest,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
            )
            return 0

        run_training(
            experiment_path,
            manifest,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    except RuntimeError as exc:
        error_text = str(exc).lower()
        if "out of memory" in error_text or "cuda" in error_text:
            print(
                "HATA: CUDA/OOM benzeri bir hata olustu. "
                "Batch size degerini 1'e dusurmeyi veya grad accumulation degerini artirmayi deneyin.",
                file=sys.stderr,
            )
        print(f"Detay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            "HATA: ImportError olustu. Coqui TTS, trainer, PyTorch veya CUDA kurulumu eksik olabilir.",
            file=sys.stderr,
        )
        print(f"Detay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except TypeError as exc:
        print(
            "HATA: Coqui training config API'si beklenen argumanlari kabul etmedi.",
            file=sys.stderr,
        )
        print(f"Detay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except TrainingError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Training tamamlandi veya trainer sureci normal bitti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
