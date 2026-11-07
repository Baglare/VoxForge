# -*- coding: utf-8 -*-
"""Pick the PyTorch CUDA wheel index that really hosts the installed torch version (VF-015).

Runs with the isolated env's own python (standard library only):

    <env>\\Scripts\\python.exe scripts/benchmark/torch_cuda.py --torch-version 2.15.0

Why: PyPI only ships CPU torch for Windows. The old setup script mapped every torch >= 2.6
to ``cu126``; torch 2.15.0 is not published there (only cu132/cu134), the forced
reinstall failed and the env silently stayed on ``+cpu``. Here the index is chosen from
what download.pytorch.org actually lists for this Python/platform, limited to the CUDA
version the installed NVIDIA driver supports. Prints one JSON object; exit 1 if no index.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import sysconfig
import urllib.request
from typing import Iterable

INDEX_ROOT = "https://download.pytorch.org/whl"
# Newest first. Unknown future tags are simply never tried; pass --index to force one.
KNOWN_INDEXES = ("cu134", "cu132", "cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118")


def parse_driver_cuda(nvidia_smi_output: str) -> tuple[int, int] | None:
    """Highest CUDA runtime the driver supports, from ``nvidia-smi`` header text."""
    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def index_cuda(tag: str) -> tuple[int, int]:
    """``cu126`` -> (12, 6); ``cu134`` -> (13, 4)."""
    digits = tag[2:]
    return int(digits[:-1]), int(digits[-1])


def wheel_tags() -> tuple[str, str]:
    """(python tag, platform tag) of the running interpreter, e.g. ('cp312', 'win_amd64')."""
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return python_tag, sysconfig.get_platform().replace("-", "_").replace(".", "_")


def wheel_name(package: str, version: str, index: str, python_tag: str, platform_tag: str) -> str:
    return f"{package}-{version}+{index}-{python_tag}-{python_tag}-{platform_tag}.whl"


def choose_index(
    version: str,
    driver: tuple[int, int] | None,
    listings: dict[str, str],
    python_tag: str,
    platform_tag: str,
    candidates: Iterable[str] = KNOWN_INDEXES,
) -> str | None:
    """Newest CUDA index that hosts ``torch==version`` for this interpreter and the driver can run."""
    for index in candidates:
        if driver is not None and index_cuda(index) > driver:
            continue
        if wheel_name("torch", version, index, python_tag, platform_tag) in listings.get(index, ""):
            return index
    return None


def fetch_listing(index: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(f"{INDEX_ROOT}/{index}/torch/", timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", "replace")


def driver_cuda() -> tuple[tuple[int, int] | None, str | None]:
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"nvidia-smi çalışmadı: {exc}"
    found = parse_driver_cuda(output)
    return found, None if found else "nvidia-smi çıktısında CUDA sürümü bulunamadı"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--torch-version", required=True, help="ör. 2.15.0 (yerel +cpu eki olmadan)")
    parser.add_argument("--index", default="", help="İndeksi zorla (ör. cu132); yine de varlığı kontrol edilir.")
    args = parser.parse_args(argv)

    version = args.torch_version.split("+")[0]
    python_tag, platform_tag = wheel_tags()
    driver, driver_error = driver_cuda()
    candidates = (args.index,) if args.index else KNOWN_INDEXES
    listings: dict[str, str] = {}
    errors: dict[str, str] = {}
    for index in candidates:
        if driver is not None and index_cuda(index) > driver and not args.index:
            continue
        try:
            listings[index] = fetch_listing(index)
        except Exception as exc:  # noqa: BLE001 - reported, other indexes still tried
            errors[index] = f"{type(exc).__name__}: {exc}"
    chosen = choose_index(version, None if args.index else driver, listings, python_tag, platform_tag, candidates)
    report = {
        "torch_version": version,
        "python_tag": python_tag,
        "platform_tag": platform_tag,
        "driver_cuda": f"{driver[0]}.{driver[1]}" if driver else None,
        "driver_error": driver_error,
        "checked": sorted(listings),
        "fetch_errors": errors,
        "index": chosen,
        "index_url": f"{INDEX_ROOT}/{chosen}" if chosen else None,
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
