# -*- coding: utf-8 -*-
"""Read-only VoxForge workspace / disk report (VF-016). Deletes nothing.

    py -3 scripts/workspace_report.py            # Markdown to stdout
    py -3 scripts/workspace_report.py --json out.json

- real sizes: symlinks/junctions are listed but NOT followed (outputs/, profiles/, .venv
  in the repo are junctions into C:\\AvenoxData), hard links counted once;
- every git worktree: clean / dirty (ignored files listed separately: they can hold
  benchmark evidence), whether its branch tip is the head of a merged GitHub PR
  (``gh`` when available; squash merges make ``git branch --merged`` useless here);
- VoxForge environments, model caches and the shared user caches (pip, uv, HF) that
  usually explain a quickly shrinking disk.

Removal decisions stay with docs/WORKSPACE_HYGIENE.md and a human/agent check; the
script only reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AVENOX_DATA_ROOT") or Path(REPO_ROOT.anchor) / "AvenoxData")
HOME = Path.home()
LOCAL = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))


def is_link(entry: os.DirEntry) -> bool:
    return entry.is_symlink() or bool(getattr(entry, "is_junction", lambda: False)())


def real_size(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """Bytes under ``path`` without following links; hard links once."""
    seen = set() if seen is None else seen
    total = 0
    try:
        entries = list(os.scandir(path))
    except OSError:
        return 0
    for entry in entries:
        try:
            if is_link(entry):
                continue
            if entry.is_dir(follow_symlinks=False):
                total += real_size(Path(entry.path), seen)
                continue
            stat = entry.stat(follow_symlinks=False)
            key = (stat.st_dev, stat.st_ino)
            if stat.st_ino and key in seen:
                continue
            seen.add(key)
            total += stat.st_size
        except OSError:
            continue
    return total


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else ""


def merged_pr_heads() -> dict[str, int]:
    if shutil.which("gh") is None:
        return {}
    result = subprocess.run(
        ["gh", "pr", "list", "--state", "merged", "--limit", "200", "--json", "number,headRefOid"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return {item["headRefOid"]: item["number"] for item in json.loads(result.stdout or "[]")}
    except json.JSONDecodeError:
        return {}


def worktrees() -> list[dict[str, Any]]:
    heads = merged_pr_heads()
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in git("worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if current:
                rows.append(current)
            current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value or True
    for row in rows:
        path = Path(row["worktree"])
        status = git("status", "--porcelain", cwd=path) if path.is_dir() else "?"
        ignored = [line[3:] for line in git("status", "--porcelain", "--ignored", cwd=path).splitlines() if line.startswith("!! ")]
        head = row.get("HEAD", "")
        row.update(
            path=str(path),
            branch=str(row.get("branch", "")).replace("refs/heads/", ""),
            clean=status == "",
            ignored=ignored,
            merged_pr=heads.get(head),
            size_bytes=real_size(path) if path.is_dir() else 0,
            main=path.resolve() == REPO_ROOT.resolve(),
        )
    return rows


def sizes(paths: dict[str, Path]) -> dict[str, int]:
    return {name: real_size(path) for name, path in paths.items() if path.is_dir()}


def gb(value: int) -> str:
    return f"{value / 2**30:.2f} GB"


def report() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT.anchor)
    env_root = DATA_ROOT / "Environments"
    return {
        "disk": {"free": usage.free, "total": usage.total},
        "worktrees": worktrees(),
        "voxforge_data": sizes({p.name: p for p in (DATA_ROOT / "Projects" / "VoxForge").glob("*") if p.is_dir()}),
        "environments": sizes({p.name: p for p in env_root.glob("VoxForge*") if p.is_dir()}),
        "models": sizes({"HF": DATA_ROOT / "Models" / "HF", "Coqui": DATA_ROOT / "Models" / "Coqui"}),
        "user_caches": sizes({
            "pip": LOCAL / "pip" / "cache",
            "uv": LOCAL / "uv" / "cache",
            "huggingface (~/.cache, HF_HOME dışı)": HOME / ".cache" / "huggingface",
            "torch hub": HOME / ".cache" / "torch",
        }),
    }


def markdown(data: dict[str, Any]) -> str:
    lines = [f"# VoxForge çalışma alanı raporu\n\nBoş alan: **{gb(data['disk']['free'])}** / {gb(data['disk']['total'])}\n", "## Worktree'ler"]
    for row in data["worktrees"]:
        state = "temiz" if row["clean"] else "KİRLİ"
        merged = f"PR #{row['merged_pr']} birleşmiş" if row["merged_pr"] else "birleşmiş PR head'i değil"
        role = " (ana kopya)" if row["main"] else ""
        lines.append(f"- `{row['path']}`{role} · `{row['branch']}` · {state} · {merged} · {gb(row['size_bytes'])}"
                     + (f" · yok sayılan: {', '.join(row['ignored'][:5])}" if row["ignored"] else ""))
    for title, key in (("VoxForge verisi", "voxforge_data"), ("Ortamlar", "environments"), ("Modeller", "models"), ("Paylaşılan kullanıcı önbellekleri", "user_caches")):
        lines.append(f"\n## {title}")
        for name, value in sorted(data[key].items(), key=lambda item: -item[1]):
            lines.append(f"- {name}: {gb(value)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    data = report()
    if args.json:
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
