# -*- coding: utf-8 -*-
"""Deneysel XTTS-v2 GPT encoder fine-tuning baslatma scripti."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = "experiment_manifest.json"
CHECKPOINT_DIR_NAME = "checkpoints"
TRAINING_OUTPUT_DIR_NAME = "training_output"

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
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"experiment_manifest.json okunamadi: {exc}") from exc


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
        missing.append(f"trainer.Trainer / TrainerArgs: {exc}")
    else:
        api["Trainer"] = Trainer
        api["TrainerArgs"] = TrainerArgs

    try:
        from TTS.config.shared_configs import BaseDatasetConfig
    except ImportError:
        try:
            from TTS.tts.configs.shared_configs import BaseDatasetConfig
        except ImportError as exc:
            missing.append(f"BaseDatasetConfig: {exc}")
        else:
            api["BaseDatasetConfig"] = BaseDatasetConfig
    else:
        api["BaseDatasetConfig"] = BaseDatasetConfig

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
    except ImportError as exc:
        missing.append(f"XTTS training importlari: {exc}")
    else:
        api["XttsConfig"] = XttsConfig
        api["load_tts_samples"] = load_tts_samples
        api["Xtts"] = Xtts
        api["XttsArgs"] = XttsArgs
        api["XttsAudioConfig"] = XttsAudioConfig

    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise TrainingError(
            "Coqui TTS / trainer importlari eksik veya mevcut paket API'si farkli.\n"
            f"{detail}\n"
            "Kurulu coqui-tts surumunu ve TTS official XTTS GPT training recipe'sini kontrol edin."
        )

    return api


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


def ensure_xtts_files(checkpoint_dir: Path) -> dict[str, Path]:
    """Gerekli XTTS dosyalari yoksa checkpoints klasorune indirir."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: checkpoint_dir / name for name in XTTS_FILE_URLS}

    for name, url in XTTS_FILE_URLS.items():
        target_path = paths[name]
        if target_path.is_file():
            print(f"Checkpoint mevcut: {target_path}")
            continue
        download_file(url, target_path)

    return paths


def make_dataset_config(api: dict[str, Any], experiment_path: Path, manifest: dict[str, Any]) -> Any:
    """Coqui LJSpeech dataset config nesnesini olusturur."""
    dataset_path = experiment_path / "dataset"
    eval_metadata_path = dataset_path / "metadata_eval.csv"
    kwargs = {
        "formatter": "ljspeech",
        "dataset_name": manifest["run_slug"],
        "path": str(dataset_path),
        "meta_file_train": "metadata_train.csv",
        "language": manifest.get("language", "tr"),
    }
    if eval_metadata_path.is_file():
        kwargs["meta_file_val"] = "metadata_eval.csv"

    BaseDatasetConfig = api["BaseDatasetConfig"]
    try:
        return BaseDatasetConfig(**kwargs)
    except TypeError:
        kwargs.pop("meta_file_val", None)
        try:
            return BaseDatasetConfig(**kwargs)
        except TypeError:
            kwargs.pop("dataset_name", None)
            return BaseDatasetConfig(**kwargs)


def accepts_argument(callable_object: Any, argument_name: str) -> bool:
    """Bir sinif/fonksiyonun arguman kabul edip etmedigini guvenli kontrol eder."""
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return True
    return argument_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def make_trainer_args(TrainerArgs: Any, max_steps: int, grad_accum: int, start_with_eval: bool) -> Any:
    """TrainerArgs nesnesini surum farklarina toleransli sekilde olusturur."""
    kwargs = {
        "restore_path": None,
        "skip_train_epoch": False,
        "start_with_eval": start_with_eval,
        "grad_accum_steps": grad_accum,
    }
    if accepts_argument(TrainerArgs, "max_steps"):
        kwargs["max_steps"] = max_steps

    trainer_args = TrainerArgs(**kwargs)
    if hasattr(trainer_args, "max_steps"):
        setattr(trainer_args, "max_steps", max_steps)
    return trainer_args


def make_xtts_config(
    api: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint_paths: dict[str, Path],
    output_path: Path,
    batch_size: int,
    max_steps: int,
) -> Any:
    """XTTS config nesnesini kucuk deneysel varsayilanlarla olusturur."""
    XttsArgs = api["XttsArgs"]
    XttsAudioConfig = api["XttsAudioConfig"]
    XttsConfig = api["XttsConfig"]

    model_args = XttsArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        dvae_checkpoint=str(checkpoint_paths["dvae.pth"]),
        xtts_checkpoint=str(checkpoint_paths["model.pth"]),
        tokenizer_file=str(checkpoint_paths["vocab.json"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    return XttsConfig(
        output_path=str(output_path),
        model_args=model_args,
        run_name=f"{manifest['run_slug']}_gpt",
        project_name="VoxForge_XTTS_GPT_Experiment",
        run_description="Experimental local XTTS-v2 GPT encoder fine-tuning.",
        audio=audio_config,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        batch_group_size=16,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        eval_split_max_size=256,
        print_step=25,
        plot_step=100,
        save_step=max(50, min(max_steps, 100)),
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [max_steps + 1], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )


def load_samples(api: dict[str, Any], dataset_config: Any, manifest: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Coqui dataset loader ile train/eval sample listelerini yukler."""
    eval_exists = int(manifest.get("eval_samples", 0)) > 0
    load_tts_samples = api["load_tts_samples"]
    samples = load_tts_samples(
        dataset_config,
        eval_split=eval_exists,
        eval_split_max_size=max(1, int(manifest.get("eval_samples", 1))),
        eval_split_size=0.10,
    )
    if isinstance(samples, tuple):
        return samples[0], samples[1]
    return samples, []


def run_training(experiment_path: Path, manifest: dict[str, Any], max_steps: int, batch_size: int, grad_accum: int) -> None:
    """Coqui Trainer ile deneysel XTTS GPT training baslatir."""
    api = import_training_api()
    checkpoint_paths = ensure_xtts_files(experiment_path / CHECKPOINT_DIR_NAME)
    output_path = experiment_path / TRAINING_OUTPUT_DIR_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_config = make_dataset_config(api, experiment_path, manifest)
    config = make_xtts_config(api, manifest, checkpoint_paths, output_path, batch_size, max_steps)
    config.datasets = [dataset_config]

    train_samples, eval_samples = load_samples(api, dataset_config, manifest)
    print(f"Coqui train samples: {len(train_samples)}")
    print(f"Coqui eval samples: {len(eval_samples)}")

    model = api["Xtts"].init_from_config(config)
    try:
        model.load_checkpoint(
            config,
            checkpoint_path=str(checkpoint_paths["model.pth"]),
            vocab_path=str(checkpoint_paths["vocab.json"]),
            eval=False,
            strict=False,
        )
    except TypeError:
        model.load_checkpoint(
            config,
            checkpoint_path=str(checkpoint_paths["model.pth"]),
            vocab_path=str(checkpoint_paths["vocab.json"]),
            strict=False,
        )

    trainer_args = make_trainer_args(
        api["TrainerArgs"],
        max_steps=max_steps,
        grad_accum=grad_accum,
        start_with_eval=bool(eval_samples),
    )
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
        help="Manifest ve ortam bilgisini yazdir, training baslatma.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    experiment_path = resolve_experiment_path(args.experiment)

    try:
        manifest = read_manifest(experiment_path)
        print_preflight(
            experiment_path,
            manifest,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        if args.dry_run:
            print("Dry-run modu: training baslatilmadi.")
            return 0

        run_training(
            experiment_path,
            manifest,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
            print(
                "HATA: CUDA/OOM benzeri bir hata olustu. "
                "Batch size degerini 1'e dusurmeyi veya grad accumulation degerini artirmayi deneyin.",
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
