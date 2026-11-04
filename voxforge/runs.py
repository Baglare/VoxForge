"""Run identity, atomic JSON writes and training checkpoint provenance.

VF-002: experiment and generation records must not silently mix runs.
- Every new record gets a ``run_id`` and ``schema_version``.
- Training success only counts checkpoints created or changed by *this* run.
- Evaluators prefer the run manifest; legacy folder scans are flagged.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
RUN_MANIFEST_DIR_NAME = "run_manifests"
TRAINING_RUN_KIND = "xtts_gpt_training"

CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
CHECKPOINT_MARKERS = ("checkpoint",)


class RunRecordError(RuntimeError):
    """Raised when a run record is missing or inconsistent."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(now: datetime | None = None) -> str:
    """Sortable, collision resistant id: ``YYYYmmddTHHMMSSZ-<6 hex>``."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON via ``.partial`` + ``os.replace`` so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".partial")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def is_checkpoint_file(path: Path) -> bool:
    lower_name = path.name.lower()
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return True
    return any(marker in lower_name for marker in CHECKPOINT_MARKERS)


def snapshot_files(
    root: Path,
    predicate: Callable[[Path], bool] = is_checkpoint_file,
) -> dict[str, dict[str, int]]:
    """Map ``relative path -> {size, mtime_ns}`` for matching files under root."""
    if not root.is_dir():
        return {}
    snapshot: dict[str, dict[str, int]] = {}
    for path in root.rglob("*"):
        if path.is_file() and predicate(path):
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            snapshot[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return snapshot


def changed_since(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> list[str]:
    """Relative paths that are new or whose size/mtime changed."""
    return sorted(
        relative
        for relative, info in after.items()
        if before.get(relative) != info
    )


def run_manifest_dir(experiment_path: Path) -> Path:
    return experiment_path / RUN_MANIFEST_DIR_NAME


def start_training_run(experiment_path: Path, training_output: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Create the manifest before ``trainer.fit()`` with a checkpoint inventory."""
    run_id = new_run_id()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRAINING_RUN_KIND,
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "training_output": str(training_output),
        "settings": settings,
        "preexisting_checkpoints": snapshot_files(training_output),
        "checkpoints": [],
        "warnings": [],
        "error": None,
    }
    write_json_atomic(run_manifest_dir(experiment_path) / f"{run_id}.json", manifest)
    return manifest


def finish_training_run(
    experiment_path: Path,
    manifest: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    """Record only checkpoints produced by this run; none means failure."""
    training_output = Path(manifest["training_output"])
    after = snapshot_files(training_output)
    produced = changed_since(manifest["preexisting_checkpoints"], after)
    manifest["checkpoints"] = [
        {"path": relative, **after[relative]} for relative in produced
    ]
    manifest["finished_at"] = utc_now_iso()
    if error:
        manifest["status"] = "failed"
        manifest["error"] = error
    elif not produced:
        manifest["status"] = "failed"
        manifest["error"] = "Bu kosu yeni veya degismis checkpoint uretmedi."
        if manifest["preexisting_checkpoints"]:
            manifest["warnings"].append(
                "training_output icinde onceki kosulardan checkpoint var; basari sayilmadi."
            )
    else:
        manifest["status"] = "succeeded"
    write_json_atomic(run_manifest_dir(experiment_path) / f"{manifest['run_id']}.json", manifest)
    return manifest


def read_run_manifest(experiment_path: Path, run_id: str) -> dict[str, Any]:
    path = run_manifest_dir(experiment_path) / f"{run_id}.json"
    if not path.is_file():
        raise RunRecordError(f"Run manifest bulunamadi: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def has_run_manifests(experiment_path: Path) -> bool:
    """Any manifest file (even failed/unreadable) means this is not a legacy experiment."""
    directory = run_manifest_dir(experiment_path)
    return directory.is_dir() and any(directory.glob("*.json"))


def latest_successful_run(experiment_path: Path) -> dict[str, Any] | None:
    directory = run_manifest_dir(experiment_path)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("kind") == TRAINING_RUN_KIND and manifest.get("status") == "succeeded":
            return manifest
    return None


def training_checkpoint_candidates(
    experiment_path: Path,
    training_output: Path,
    run_id: str | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Checkpoint files to choose from plus provenance for reports.

    With a run id (or the latest successful manifest) only that run's checkpoints
    are returned. If manifests exist but none succeeded, selection fails instead
    of falling back. The flagged legacy scan is used only when no manifest exists.
    """
    if run_id:
        manifest = read_run_manifest(experiment_path, run_id)
        if manifest.get("status") != "succeeded":
            raise RunRecordError(f"Run basarili degil ({manifest.get('status')}): {run_id}")
    else:
        manifest = latest_successful_run(experiment_path)
        if manifest is None and has_run_manifests(experiment_path):
            raise RunRecordError(
                "Run manifestleri var ama basarili egitim kosusu yok; eski checkpointlere "
                "otomatik gecilmedi. Gerekirse --run-id ile basarili bir kosu secin."
            )

    if manifest is None:
        paths = sorted(
            (path for path in training_output.rglob("*") if path.is_file() and is_checkpoint_file(path)),
            key=lambda item: str(item).lower(),
        ) if training_output.is_dir() else []
        return paths, {
            "source": "legacy_discovery",
            "run_id": None,
            "warning": (
                "Run manifest yok; checkpoint klasor taramasiyla secildi. "
                "Onceki kosulardan kalan dosyalar secilmis olabilir."
            ),
        }

    root = Path(manifest["training_output"])
    paths = [root / item["path"] for item in manifest.get("checkpoints", [])]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RunRecordError("Run manifestindeki checkpoint dosyalari eksik: " + ", ".join(missing))
    return paths, {"source": "run_manifest", "run_id": manifest["run_id"], "warning": None}


_NUMBERED = re.compile(r"(best_model|checkpoint)_(\d+)\.pth")


def numbered(path: Path, prefix: str) -> int | None:
    match = _NUMBERED.fullmatch(path.name)
    if match and match.group(1) == prefix:
        return int(match.group(2))
    return None


def wav_part_status(path: Path) -> dict[str, Any]:
    """Check one generated WAV part: exists, readable, non-empty."""
    status: dict[str, Any] = {"path": str(path), "ok": False, "error": None, "frames": 0}
    if not path.is_file():
        status["error"] = "dosya yok"
        return status
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        status["error"] = f"okunamadi: {exc}"
        return status
    status["frames"] = frames
    if frames <= 0:
        status["error"] = "bos ses"
        return status
    status["ok"] = True
    return status
