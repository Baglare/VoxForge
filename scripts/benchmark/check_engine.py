# -*- coding: utf-8 -*-
"""Light engine check without downloading any model (VF-Q1).

    <chatterbox-venv>\\python.exe scripts/benchmark/check_engine.py --engine chatterbox

Verifies, using only the installed code:
- chatterbox-tts comes from the pinned GitHub commit (not the PyPI build without V3);
- ChatterboxMultilingualTTS.from_pretrained accepts ``t3_model`` and the name "v3"
  resolves to a V3 checkpoint file (no network, no weights);
- Turkish ("tr") is a supported language;
- torch sees CUDA.
Prints a JSON summary; exit code 1 if anything fails.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sources import chatterbox_pin_problem, describe, load_pins, source_revision  # noqa: E402


def check_chatterbox() -> dict:
    pin = load_pins()["chatterbox"]
    report: dict = {"engine": "chatterbox", "pinned": f"{pin['repo']}@{pin['commit']}", "checks": {}, "ok": False}
    checks = report["checks"]

    installed = source_revision(pin["package"])
    checks["source"] = describe(installed)
    problem = chatterbox_pin_problem(installed, pin)
    checks["source_ok"] = problem is None
    if problem:
        report["error"] = problem

    perth = source_revision("resemble-perth")
    checks["resemble_perth_source"] = describe(perth)

    try:
        import chatterbox.mtl_tts as mtl
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"chatterbox.mtl_tts içe aktarılamadı: {type(exc).__name__}: {exc}"
        return report

    parameters = inspect.signature(mtl.ChatterboxMultilingualTTS.from_pretrained).parameters
    checks["t3_model_parameter"] = "t3_model" in parameters
    resolver = getattr(mtl, "_resolve_multilingual_t3_model", None)
    try:
        resolved = resolver("v3") if resolver else getattr(mtl, "MULTILINGUAL_T3_MODELS", {}).get("v3")
    except Exception as exc:  # noqa: BLE001
        resolved = None
        checks["v3_resolve_error"] = f"{type(exc).__name__}: {exc}"
    checks["v3_checkpoint"] = resolved
    checks["v3_ok"] = bool(resolved) and "v3" in str(resolved)
    languages = getattr(mtl, "SUPPORTED_LANGUAGES", {})
    checks["turkish_supported"] = "tr" in languages

    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = bool(torch.cuda.is_available())
        checks["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        checks["cuda_available"] = False
        checks["torch_error"] = f"{type(exc).__name__}: {exc}"

    required = ("source_ok", "t3_model_parameter", "v3_ok", "turkish_supported", "cuda_available")
    report["ok"] = all(checks.get(name) for name in required)
    if not report["ok"] and "error" not in report:
        report["error"] = "Başarısız kontroller: " + ", ".join(name for name in required if not checks.get(name))
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, choices=("chatterbox",))
    parser.parse_args(argv)
    report = check_chatterbox()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("V3 API kontrolü: " + ("GEÇTİ" if report["ok"] else "BAŞARISIZ"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
