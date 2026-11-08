# -*- coding: utf-8 -*-
"""VF-017 local A/B listening report (HTML next to the WAVs, git ignored, never published).

    .venv\\Scripts\\python.exe scripts/benchmark/vf017_report.py --probe outputs/vf017/probe [--probe outputs/vf017/fix]

Reads each probe's manifest.json + analysis.json and writes ``outputs/vf017/DINLEME.html``:
clearly labelled players (relative paths, opens with file://) and the measured numbers
beside every sample. Numbers are measurements, not a naturalness verdict: listen.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COLUMNS = [
    ("cer", "CER"), ("sim_reference", "ECAPA→referans"), ("sim_default", "ECAPA→A (aynı metin)"),
    ("overall_rate", "hece/sn"), ("articulation_rate", "artikülasyon"), ("pause_seconds", "duraklama sn"),
    ("head_vs_rest", "ilk2/kalan"), ("f0_range_st", "F0 aralığı (yarım ton)"), ("duration", "süre sn"),
]


def row_html(name: str, wav: Path, base: Path, item: dict, title: str) -> str:
    rel = os.path.relpath(wav, base).replace(os.sep, "/")
    cells = "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key, _ in COLUMNS)
    return (f"<tr><td><b>{html.escape(title)}</b><br><code>{html.escape(name)}</code><br>"
            f"<small>{html.escape(item.get('label') or '')}</small></td>"
            f"<td><audio controls preload=none src=\"{html.escape(rel)}\"></audio></td>{cells}</tr>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "vf017" / "DINLEME.html")
    parser.add_argument("--titles", type=Path, help="JSON {ad: başlık} (isteğe bağlı)")
    args = parser.parse_args(argv)
    titles = json.loads(args.titles.read_text(encoding="utf-8")) if args.titles else {}
    base = args.out.parent
    sections: dict[str, list[str]] = {}
    for probe in args.probe:
        manifest = json.loads((probe / "manifest.json").read_text(encoding="utf-8"))
        analysis = json.loads((probe / "analysis.json").read_text(encoding="utf-8"))
        for row in manifest:
            item = analysis.get(row["name"], {})
            group = titles.get(f"group:{row['group']}", row["group"])
            sections.setdefault(group, []).append(row_html(row["name"], Path(row["wav"]), base, item, titles.get(row["name"], row["name"])))
    head = "".join(f"<th>{label}</th>" for _, label in COLUMNS)
    body = "".join(
        f"<h2>{html.escape(group)}</h2><table><tr><th>örnek</th><th>dinle</th>{head}</tr>{''.join(rows)}</table>"
        for group, rows in sections.items()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "<!doctype html><meta charset=utf-8><title>VF-017 dinleme</title>"
        "<style>body{font:14px system-ui;margin:16px;max-width:1400px}table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top;font-size:13px}audio{width:260px}</style>"
        "<h1>VF-017 — VoxCPM2 anlatım A/B dinleme</h1>"
        "<p>Aynı mimo referansı, CFG 2.0, 10 adım, tohum 1234. Sayılar ölçümdür, doğallık kararı değildir: kulaklıkla dinleyin. "
        "ECAPA kısa kliplerde gürültülüdür; yalnız aynı metin içinde karşılaştırın. Ayrıntı: docs/VF017_NARRATION_FINDINGS.md.</p>"
        + body,
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
