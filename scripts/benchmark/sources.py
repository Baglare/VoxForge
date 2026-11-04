# -*- coding: utf-8 -*-
"""Where benchmark engine packages came from (VF-Q1). Standard library only.

A package version is not enough: chatterbox-tts reports 0.1.7 both on PyPI (no V3)
and on GitHub master (V3). pip records the real source in ``direct_url.json``
(PEP 610); that revision is what results.json stores.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

PINS_PATH = Path(__file__).with_name("engine_pins.json")


def load_pins() -> dict[str, Any]:
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


def parse_direct_url(text: str | None) -> dict[str, Any]:
    """PEP 610 direct_url.json -> {"source": "git"|"url"|"local"|"index", "url", "commit"}."""
    if not text:
        return {"source": "index", "url": None, "commit": None}
    data = json.loads(text)
    vcs = data.get("vcs_info")
    if vcs:
        return {
            "source": vcs.get("vcs", "vcs"),
            "url": data.get("url"),
            "commit": vcs.get("commit_id"),
            "requested_revision": vcs.get("requested_revision"),
        }
    if data.get("dir_info") is not None:
        return {"source": "local", "url": data.get("url"), "commit": None, "editable": data["dir_info"].get("editable", False)}
    return {"source": "url", "url": data.get("url"), "commit": None}


def source_revision(package: str) -> dict[str, Any] | None:
    """Installed source of a distribution, or None if it is not installed."""
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None
    info = parse_direct_url(dist.read_text("direct_url.json"))
    info["version"] = dist.version
    return info


def describe(info: dict[str, Any] | None) -> str:
    if info is None:
        return "kurulu değil"
    if info.get("commit"):
        return f"{info['url']}@{info['commit']} (sürüm {info['version']})"
    if info["source"] == "index":
        return f"PyPI/indeks, sürüm {info['version']}"
    return f"{info['source']}: {info.get('url')} (sürüm {info['version']})"


def chatterbox_pin_problem(info: dict[str, Any] | None, pin: dict[str, Any]) -> str | None:
    """None if the installed chatterbox-tts is exactly the pinned GitHub commit."""
    if info is None:
        return "chatterbox-tts kurulu değil."
    if info.get("commit") != pin["commit"]:
        return (
            f"chatterbox-tts sabitlenen kaynaktan değil: kurulu {describe(info)}; beklenen "
            f"{pin['repo']}@{pin['commit']}. Ortamı yenileyin: scripts\\benchmark\\setup_env.ps1 -Engine chatterbox"
        )
    return None
