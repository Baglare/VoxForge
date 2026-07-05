# -*- coding: utf-8 -*-
"""XTTS fine-tuned checkpoint icin inference parametre sweep denemesi."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_xtts_finetuned_checkpoint import (
    CHUNK_MAX_CHARS,
    PROJECT_ROOT,
    TRAINING_OUTPUT_DIR_NAME,
    EvaluationError,
    import_xtts_api,
    init_model,
    load_config,
    load_model_checkpoint,
    move_model_to_device,
    output_sample_rate,
    patch_config_paths,
    resolve_base_files,
    resolve_project_path,
    resolve_speaker_wav,
    save_wav_from_synthesis,
    signature_text,
)

try:
    from audio_concat_utils import concatenate_wavs
    from text_chunking_utils import split_text_for_tts, summarize_chunks
except ImportError:
    from scripts.audio_concat_utils import concatenate_wavs
    from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks


LANGUAGE = "tr"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "finetuned_eval" / "param_sweep"
REPORT_JSON_PATH = PROJECT_ROOT / "outputs" / "reports" / "inference_param_sweep_report.json"
REPORT_MD_PATH = PROJECT_ROOT / "outputs" / "reports" / "inference_param_sweep_report.md"
SUPPORTED_VARIANTS = ("best_model", "best_model_72", "checkpoint_71", "base")

TEST_TEXTS = [
    "Merhaba, bu kısa bir kesilme kontrolüdür.",
    (
        "Bugünkü denemede aynı checkpoint için farklı inference ayarlarını dinleyip "
        "cümlenin sonuna kadar gelip gelmediğini kontrol ediyoruz."
    ),
    (
        "Bu daha uzun paragraf, checkpoint_71 çıktılarında cümlenin erken kesilip kesilmediğini "
        "anlamak için hazırlandı. Metin doğal bir akışla birkaç cümle içeriyor, konuşmanın sonunda "
        "sesin aniden durmaması ve son kelimelerin anlaşılır kalması bekleniyor."
    ),
    "Bu ayarlarla üretilen soru cümlesi kulağa daha doğal geliyor mu?",
]

PARAM_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "conservative": {"temperature": 0.65, "top_p": 0.8, "top_k": 50},
    "stable": {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 50,
        "repetition_penalty": 5.0,
    },
    "longer_attempt": {
        "temperature": 0.75,
        "top_p": 0.9,
        "top_k": 80,
        "length_penalty": 1.0,
    },
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuned XTTS checkpoint icin inference parametre sweep calistirir."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment klasoru. Ornek: experiments/baglare-xtts-exp01",
    )
    parser.add_argument(
        "--variant",
        default="checkpoint_71",
        choices=SUPPORTED_VARIANTS,
        help="Test edilecek varyant. Varsayilan: checkpoint_71",
    )
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Opsiyonel referans ses yolu.",
    )
    return parser.parse_args(argv)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def find_best_model(training_output: Path) -> Path | None:
    return latest_file([path for path in training_output.rglob("best_model.pth") if path.is_file()])


def find_named_checkpoint(training_output: Path, filename: str) -> Path | None:
    return latest_file([path for path in training_output.rglob(filename) if path.is_file()])


def find_checkpoint_71(training_output: Path) -> Path | None:
    exact_match = find_named_checkpoint(training_output, "checkpoint_71.pth")
    if exact_match:
        return exact_match

    candidates: list[Path] = []
    for path in training_output.rglob("checkpoint_*.pth"):
        if not path.is_file():
            continue
        match = re.fullmatch(r"checkpoint_(\d+)\.pth", path.name)
        if match and int(match.group(1)) == 71:
            candidates.append(path)
    return latest_file(candidates)


def resolve_variant_checkpoint(
    experiment_path: Path,
    base_files: dict[str, Path],
    variant: str,
) -> Path:
    if variant == "base":
        return base_files["model.pth"].resolve()

    training_output = experiment_path / TRAINING_OUTPUT_DIR_NAME
    if not training_output.is_dir():
        raise EvaluationError(f"training_output klasoru bulunamadi: {training_output}")

    variant_paths = {
        "best_model": find_best_model(training_output),
        "best_model_72": find_named_checkpoint(training_output, "best_model_72.pth"),
        "checkpoint_71": find_checkpoint_71(training_output),
    }
    checkpoint_path = variant_paths.get(variant)
    if checkpoint_path is None:
        raise EvaluationError(f"{variant} checkpoint bulunamadi: {training_output}")
    return checkpoint_path.resolve()


def load_ready_model(
    label: str,
    config_class: Any,
    model_class: Any,
    base_files: dict[str, Path],
    checkpoint_path: Path,
) -> tuple[Any, Any]:
    print(f"{label} modeli hazirlaniyor.")
    print(f"{label} checkpoint: {checkpoint_path}")
    config = load_config(config_class, base_files["config.json"])
    patch_config_paths(config, base_files, checkpoint_path)
    model = init_model(model_class, config)
    load_model_checkpoint(model, config, base_files, checkpoint_path, label)
    move_model_to_device(model)
    return config, model


def synthesize_with_params(
    model: Any,
    config: Any,
    text: str,
    speaker_wav: Path,
    params: dict[str, Any],
) -> Any:
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
                **params,
            ),
        ),
        (
            "keyword text/config/speaker_wav/language",
            lambda: synthesize_method(
                text=text,
                config=config,
                speaker_wav=str(speaker_wav),
                language=LANGUAGE,
                **params,
            ),
        ),
        (
            "speaker_wav list",
            lambda: synthesize_method(
                text,
                config,
                speaker_wav=[str(speaker_wav)],
                language=LANGUAGE,
                **params,
            ),
        ),
    ]

    type_errors: list[str] = []
    other_errors: list[str] = []
    for label, attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            type_errors.append(f"{label}: TypeError: {exc}")
        except Exception as exc:
            other_errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if type_errors and not other_errors:
        raise TypeError(
            "XTTS synthesize parametreleri kabul etmedi:\n"
            + "\n".join(f"- {error}" for error in type_errors)
        )

    raise EvaluationError(
        "XTTS synthesize cagrisi basarisiz. Denenen varyantlar:\n"
        + "\n".join(f"- {error}" for error in [*type_errors, *other_errors])
    )


def wave_duration_seconds(wav_path: Path) -> float | None:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return None
            return wav_file.getnframes() / float(sample_rate)
    except (OSError, wave.Error):
        return None


def ffprobe_duration_seconds(wav_path: Path, errors: list[dict[str, str]]) -> tuple[float | None, str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        fallback_duration = wave_duration_seconds(wav_path)
        errors.append(
            {
                "variant": "",
                "param_set": "",
                "test": "",
                "message": "ffprobe bulunamadi; sure icin wave fallback kullanildi.",
            }
        )
        return fallback_duration, "wave_fallback"

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fallback_duration = wave_duration_seconds(wav_path)
        errors.append(
            {
                "variant": "",
                "param_set": "",
                "test": "",
                "message": f"ffprobe sure olcemedi: {result.stderr.strip()}",
            }
        )
        return fallback_duration, "wave_fallback"

    try:
        return float(result.stdout.strip()), "ffprobe"
    except ValueError:
        fallback_duration = wave_duration_seconds(wav_path)
        errors.append(
            {
                "variant": "",
                "param_set": "",
                "test": "",
                "message": f"ffprobe sure ciktisi sayi degil: {result.stdout.strip()}",
            }
        )
        return fallback_duration, "wave_fallback"


def cutoff_flags_for_output(
    text: str,
    duration: float | None,
    base_duration: float | None,
) -> list[str]:
    flags: list[str] = []
    if duration is None:
        return flags

    if len(text) >= 120 and duration < 2.0:
        flags.append("likely_cutoff")

    if base_duration is not None and base_duration > 0:
        is_much_shorter = duration < (base_duration * 0.65)
        is_at_least_one_second_shorter = (base_duration - duration) >= 1.0
        if is_much_shorter and is_at_least_one_second_shorter:
            flags.append("possibly_cutoff")

    return flags


def run_single_output(
    label: str,
    model: Any,
    config: Any,
    speaker_wav: Path,
    text: str,
    output_path: Path,
    params: dict[str, Any],
    errors: list[dict[str, str]],
) -> tuple[float | None, str, dict[str, Any]]:
    chunks = split_text_for_tts(text, max_chars=CHUNK_MAX_CHARS)
    chunk_summary = summarize_chunks(chunks)
    if not chunks:
        raise EvaluationError("TTS icin bos metin uretildi.")

    if len(chunks) == 1:
        synth_output = synthesize_with_params(
            model=model,
            config=config,
            text=chunks[0],
            speaker_wav=speaker_wav,
            params=params,
        )
        sample_rate = output_sample_rate(config, synth_output)
        save_wav_from_synthesis(synth_output, output_path, sample_rate=sample_rate)
    else:
        chunk_dir = output_path.parent / "chunks" / output_path.stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths: list[Path] = []
        print(f"{label} chunking kullaniliyor: {len(chunks)} parca")

        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"chunk_{index:02d}.wav"
            print(f"{label} chunk_{index:02d}: {len(chunk)} karakter")
            synth_output = synthesize_with_params(
                model=model,
                config=config,
                text=chunk,
                speaker_wav=speaker_wav,
                params=params,
            )
            sample_rate = output_sample_rate(config, synth_output)
            save_wav_from_synthesis(synth_output, chunk_path, sample_rate=sample_rate)
            chunk_paths.append(chunk_path)

        ok, message = concatenate_wavs(chunk_paths, output_path)
        if not ok:
            raise EvaluationError(message)
        print(f"{label} chunk birlestirme OK: {message}")

    duration, duration_source = ffprobe_duration_seconds(output_path, errors)
    print(f"{label} output: {output_path}")
    print(f"{label} duration_seconds: {duration}")
    return duration, duration_source, chunk_summary


def output_entry_key(variant: str, param_set: str, test_id: str) -> str:
    return f"{variant}/{param_set}/{test_id}"


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON rapor yazildi: {REPORT_JSON_PATH}")

    lines = [
        "# XTTS inference parameter sweep raporu",
        "",
        f"- Experiment: `{report['experiment']}`",
        f"- Tested variant: `{report['tested_variant']}`",
        f"- Selected checkpoint: `{report['selected_checkpoint']}`",
        f"- Speaker wav: `{report['speaker_wav']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Chunking used: {report['chunking_used']}",
        f"- Max chunk count: {report['chunk_count']}",
        f"- Hata sayisi: {len(report['errors'])}",
        "",
        "## Dinleme yonergesi",
        "",
        "1. Once `default` parametrelerini dinle.",
        "2. Sonra `stable` ve `longer_attempt` ciktilariyla karsilastir.",
        "3. Erken kesilme azaliyorsa o parametre seti not alinmali.",
        "4. Kalite bozuluyorsa checkpoint yerine `best_model` veya daha fazla veri tercih edilmeli.",
        "",
        "Bu otomatik kalite garantisi degildir. Erken kesilmeyi anlamak icin karsilastirmali dinleme gerekir.",
        "",
        "## Parametre setleri",
        "",
    ]

    for name, params in report["param_sets"].items():
        lines.append(f"- `{name}`: `{params}`")

    lines.extend(
        [
            "",
            "## Ciktilar",
            "",
            "| variant | param_set | test | chars | chunks | duration | flags | output |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for item in report["outputs"]:
        duration = item["duration_seconds"]
        duration_text = "-" if duration is None else f"{duration:.3f}"
        flags = ", ".join(item["cutoff_flags"]) if item["cutoff_flags"] else "-"
        lines.append(
            f"| {item['variant']} | {item['param_set']} | {item['test_id']} | "
            f"{item['char_count']} | {item.get('chunk_count', 1)} | {duration_text} | "
            f"{flags} | `{item['output_path']}` |"
        )

    lines.extend(["", "## Chunking", ""])
    for item in report["test_texts"]:
        if item["chunking_used"]:
            lines.append(f"- `{item['test_id']}`: {item['chunk_count']} parca")

    if report["errors"]:
        lines.extend(["", "## Hatalar", ""])
        for error in report["errors"]:
            lines.append(
                "- `{variant}` `{param_set}` `{test}`: {message}".format(
                    variant=error.get("variant", ""),
                    param_set=error.get("param_set", ""),
                    test=error.get("test", ""),
                    message=error.get("message", ""),
                )
            )

    lines.extend(["", "## Notlar", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown rapor yazildi: {REPORT_MD_PATH}")


def initial_report(
    experiment_path: Path,
    variant: str,
    selected_checkpoint: Path | None,
    speaker_wav: Path | None,
    output_root: Path,
    outputs: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    test_texts = []
    for index, text in enumerate(TEST_TEXTS, start=1):
        chunk_summary = summarize_chunks(split_text_for_tts(text, max_chars=CHUNK_MAX_CHARS))
        test_texts.append(
            {
                "test_id": f"test_{index:02d}",
                "text": text,
                "char_count": len(text),
                **chunk_summary,
            }
        )

    duration_seconds = {
        output_entry_key(item["variant"], item["param_set"], item["test_id"]): item["duration_seconds"]
        for item in outputs
    }
    cutoff_flags = [
        {
            "variant": item["variant"],
            "param_set": item["param_set"],
            "test_id": item["test_id"],
            "flags": item["cutoff_flags"],
        }
        for item in outputs
        if item["cutoff_flags"]
    ]
    return {
        "experiment": str(experiment_path),
        "tested_variant": variant,
        "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint else None,
        "speaker_wav": str(speaker_wav) if speaker_wav else None,
        "param_sets": PARAM_SETS,
        "test_texts": test_texts,
        "chunking_used": any(item["chunking_used"] for item in test_texts),
        "chunk_count": max((item["chunk_count"] for item in test_texts), default=0),
        "output_root": str(output_root),
        "outputs": outputs,
        "duration_seconds": duration_seconds,
        "cutoff_flags": cutoff_flags,
        "errors": errors,
        "notes": [
            "Bu otomatik kalite garantisi degildir.",
            "Erken kesilmeyi anlamak icin karsilastirmali dinleme gerekir.",
            "Daha fazla training basmadan once inference ayarlari ve checkpoint secimi degerlendirilmelidir.",
            "likely_cutoff: uzun metinde cikti 2 saniyeden kisa.",
            "possibly_cutoff: ayni testin base ciktisina gore belirgin kisa.",
        ],
    }


def run_variant_sweep(
    variant: str,
    config: Any,
    model: Any,
    speaker_wav: Path,
    output_root: Path,
    base_durations: dict[str, float | None],
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for param_set, params in PARAM_SETS.items():
        param_dir = output_root / variant / param_set
        param_dir.mkdir(parents=True, exist_ok=True)

        for index, text in enumerate(TEST_TEXTS, start=1):
            test_id = f"test_{index:02d}"
            output_path = param_dir / f"{test_id}.wav"
            label = f"{variant}/{param_set}/{test_id}"
            try:
                print(f"{label} uretiliyor...")
                duration, duration_source, chunk_summary = run_single_output(
                    label=label,
                    model=model,
                    config=config,
                    speaker_wav=speaker_wav,
                    text=text,
                    output_path=output_path,
                    params=params,
                    errors=errors,
                )
                flags = cutoff_flags_for_output(
                    text=text,
                    duration=duration,
                    base_duration=base_durations.get(test_id),
                )
                outputs.append(
                    {
                        "variant": variant,
                        "param_set": param_set,
                        "test_id": test_id,
                        "text": text,
                        "char_count": len(text),
                        "output_path": str(output_path),
                        "duration_seconds": duration,
                        "duration_source": duration_source,
                        "base_duration_seconds": base_durations.get(test_id),
                        "chunking_used": chunk_summary["chunking_used"],
                        "chunk_count": chunk_summary["chunk_count"],
                        "chunks": chunk_summary["chunks"],
                        "cutoff_flags": flags,
                    }
                )
            except TypeError as exc:
                message = f"TypeError: {exc}"
                print(f"UYARI: {label} parametre hatasi: {message}", file=sys.stderr)
                errors.append(
                    {
                        "variant": variant,
                        "param_set": param_set,
                        "test": test_id,
                        "message": message,
                    }
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"UYARI: {label} basarisiz: {message}", file=sys.stderr)
                errors.append(
                    {
                        "variant": variant,
                        "param_set": param_set,
                        "test": test_id,
                        "message": message,
                    }
                )
    return outputs


def run_base_reference(
    config: Any,
    model: Any,
    speaker_wav: Path,
    output_root: Path,
    errors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    outputs: list[dict[str, Any]] = []
    durations: dict[str, float | None] = {}
    param_dir = output_root / "base" / "default"
    param_dir.mkdir(parents=True, exist_ok=True)

    for index, text in enumerate(TEST_TEXTS, start=1):
        test_id = f"test_{index:02d}"
        output_path = param_dir / f"{test_id}.wav"
        label = f"base/default/{test_id}"
        try:
            print(f"{label} referans uretiliyor...")
            duration, duration_source, chunk_summary = run_single_output(
                label=label,
                model=model,
                config=config,
                speaker_wav=speaker_wav,
                text=text,
                output_path=output_path,
                params={},
                errors=errors,
            )
            durations[test_id] = duration
            outputs.append(
                {
                    "variant": "base",
                    "param_set": "default",
                    "test_id": test_id,
                    "text": text,
                    "char_count": len(text),
                    "output_path": str(output_path),
                    "duration_seconds": duration,
                    "duration_source": duration_source,
                    "base_duration_seconds": duration,
                    "chunking_used": chunk_summary["chunking_used"],
                    "chunk_count": chunk_summary["chunk_count"],
                    "chunks": chunk_summary["chunks"],
                    "cutoff_flags": cutoff_flags_for_output(text, duration, None),
                }
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"UYARI: {label} base referans basarisiz: {message}", file=sys.stderr)
            errors.append(
                {
                    "variant": "base",
                    "param_set": "default",
                    "test": test_id,
                    "message": message,
                }
            )
            durations[test_id] = None

    return outputs, durations


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    experiment_path = resolve_project_path(args.experiment)
    output_root = OUTPUT_ROOT / timestamp_slug()
    errors: list[dict[str, str]] = []
    outputs: list[dict[str, Any]] = []
    selected_checkpoint: Path | None = None
    speaker_wav: Path | None = None

    try:
        print("VoxForge XTTS inference parameter sweep")
        print(f"Experiment: {experiment_path}")
        print(f"Variant: {args.variant}")
        print(f"Output root: {output_root}")

        base_files = resolve_base_files(experiment_path)
        selected_checkpoint = resolve_variant_checkpoint(
            experiment_path=experiment_path,
            base_files=base_files,
            variant=args.variant,
        )
        speaker_wav = resolve_speaker_wav(experiment_path, args.speaker_wav)
        print(f"Kullanilan checkpoint: {selected_checkpoint}")
        print(f"Kullanilan speaker_wav: {speaker_wav}")

        config_class, model_class = import_xtts_api()
        base_durations: dict[str, float | None] = {}

        if args.variant != "base":
            try:
                base_config, base_model = load_ready_model(
                    label="base",
                    config_class=config_class,
                    model_class=model_class,
                    base_files=base_files,
                    checkpoint_path=base_files["model.pth"],
                )
                base_outputs, base_durations = run_base_reference(
                    config=base_config,
                    model=base_model,
                    speaker_wav=speaker_wav,
                    output_root=output_root,
                    errors=errors,
                )
                outputs.extend(base_outputs)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"UYARI: base referans uretilemedi: {message}", file=sys.stderr)
                errors.append(
                    {
                        "variant": "base",
                        "param_set": "default",
                        "test": "",
                        "message": message,
                    }
                )

        selected_config, selected_model = load_ready_model(
            label=args.variant,
            config_class=config_class,
            model_class=model_class,
            base_files=base_files,
            checkpoint_path=selected_checkpoint,
        )
        outputs.extend(
            run_variant_sweep(
                variant=args.variant,
                config=selected_config,
                model=selected_model,
                speaker_wav=speaker_wav,
                output_root=output_root,
                base_durations=base_durations,
                errors=errors,
            )
        )

        report = initial_report(
            experiment_path=experiment_path,
            variant=args.variant,
            selected_checkpoint=selected_checkpoint,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            errors=errors,
        )
        write_reports(report)

        if not any(item["variant"] == args.variant for item in outputs):
            print("HATA: Secilen varyant icin hic ses ciktisi uretilemedi.", file=sys.stderr)
            return 1

        print("Inference parameter sweep tamamlandi.")
        return 0
    except EvaluationError as exc:
        errors.append({"variant": args.variant, "param_set": "", "test": "", "message": str(exc)})
        report = initial_report(
            experiment_path=experiment_path,
            variant=args.variant,
            selected_checkpoint=selected_checkpoint,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            errors=errors,
        )
        write_reports(report)
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append({"variant": args.variant, "param_set": "", "test": "", "message": message})
        report = initial_report(
            experiment_path=experiment_path,
            variant=args.variant,
            selected_checkpoint=selected_checkpoint,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            errors=errors,
        )
        write_reports(report)
        print(f"HATA: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
