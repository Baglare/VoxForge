# -*- coding: utf-8 -*-
"""VF-017 end-to-end check: the user's VF-016 scene texts through the REAL scene narrator.

    .venv\\Scripts\\python.exe scripts/benchmark/vf017_scene_check.py --out outputs/vf017/fix

Creates a NEW scene project (never touches the user's ``vf016-ilk-deneme``), lets the scene
kinds pick their (VF-017, style-free) default presets, generates every scene on the GPU with
the same profile/variant/seed and merges with the fixed level matching. Writes a manifest
for vf017_analyze.py / vf017_report.py that also lists the user's old merged file and old
scene WAVs (read only) for side-by-side listening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.audio_join import join_wavs  # noqa: E402
from voxforge.playground.prosody import BUILTIN_PRESETS, plan_chunks  # noqa: E402
from voxforge.playground.scenes import SceneNarrator, SceneProject, SceneStore, split_scenes  # noqa: E402
from voxforge.playground.service import PlaygroundService  # noqa: E402

USER_PROJECT = REPO_ROOT / "outputs" / "playground" / "scenes" / "vf016-ilk-deneme.json"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "vf017" / "fix")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    old = json.loads(USER_PROJECT.read_text(encoding="utf-8"))
    text = "\n\n".join(f"## {scene['title']}\n{scene['text']}" for scene in old["scenes"])

    service = PlaygroundService(args.out / "playground")
    narrator = SceneNarrator(service, SceneStore(args.out / "scenes"))
    project = SceneProject(name="vf017-sahne-kontrol", slug="vf017-sahne-kontrol", profile=old["profile"],
                           variant=old["variant"], seed=old["seed"], normalize=old["normalize"], scenes=split_scenes(text))
    project = narrator.store.create(project)
    rows = []
    try:
        errors = narrator.generate_all(project)
        if errors:
            raise RuntimeError(errors)
        merged = narrator.merge(project)
    finally:
        service.manager.close()
    print(json.dumps({"merged": merged, "presets": [s.preset for s in project.scenes]}, ensure_ascii=False))

    for scene, old_scene in zip(project.scenes, old["scenes"]):
        controls = BUILTIN_PRESETS[scene.preset].controls
        spoken = plan_chunks(scene.text, controls, 250, True).spoken
        rows.append({"name": f"{scene.id}_yeni_{scene.preset}", "group": "scene_new", "wav": scene.wav, "text": spoken,
                     "label": f"VF-017 varsayılanı: {BUILTIN_PRESETS[scene.preset].name}", "style": controls.style})
        old_controls = BUILTIN_PRESETS[old_scene["preset"]].controls
        rows.append({"name": f"{scene.id}_eski_{old_scene['preset']}", "group": "scene_old", "wav": old_scene["wav"],
                     "text": plan_chunks(old_scene["text"], old_controls, 250, True).spoken,
                     "label": f"VF-016 kullanıcı çıktısı: {old_scene['preset']}", "style": old_controls.style})
    whole_text = " ".join(row["text"] for row in rows if row["group"] == "scene_new")
    rows.append({"name": "birlesik_yeni", "group": "merge", "wav": merged["wav"], "text": whole_text,
                 "label": f"VF-017: stilsiz preset'ler + seviye eşleme düzeltmesi, kazançlar {merged['gains_db']} dB"})
    rows.append({"name": "birlesik_eski", "group": "merge", "wav": old["merged_wav"],
                 "text": " ".join(row["text"] for row in rows if row["group"] == "scene_old"),
                 "label": "VF-016 kullanıcı birleşik çıktısı (stilli preset'ler, seviye eşleme etkisiz)"})
    # Aynı eski sahne sesleri + yalnız düzeltilmiş birleştirme: seviye düzeltmesinin tek başına etkisi
    relevel = args.out / "birlesik_eski_sesler_yeni_seviye.wav"
    info = join_wavs([Path(s["wav"]) for s in old["scenes"]], relevel, pauses=old["scene_gap"], trim=True, match_loudness=True)
    rows.append({"name": "birlesik_eski_sesler_yeni_seviye", "group": "merge", "wav": str(relevel),
                 "text": rows[-1]["text"], "label": f"eski sahne sesleri + düzeltilmiş seviye eşleme, kazançlar {info['gains_db']} dB"})
    (args.out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
