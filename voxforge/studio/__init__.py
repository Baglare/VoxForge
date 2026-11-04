"""Persistent Studio projects: SQLite metadata, segment cache and generation queue (VF-005)."""

from __future__ import annotations

import os
from pathlib import Path

from voxforge.studio.project import ADAPTER_VERSION, ProjectSettings, StudioProject
from voxforge.studio.queue import StudioWorker, process_job, run_pending
from voxforge.studio.store import SCHEMA_VERSION, ProjectStore

STUDIO_ROOT_ENV_VAR = "VOXFORGE_STUDIO_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "ADAPTER_VERSION",
    "SCHEMA_VERSION",
    "ProjectSettings",
    "ProjectStore",
    "StudioProject",
    "StudioWorker",
    "process_job",
    "run_pending",
    "studio_root",
]


def studio_root() -> Path:
    """Projects live outside Git: ``VOXFORGE_STUDIO_ROOT`` or ``<repo>/studio_projects``."""
    configured = os.environ.get(STUDIO_ROOT_ENV_VAR)
    return Path(configured) if configured else PROJECT_ROOT / "studio_projects"
