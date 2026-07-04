# -*- coding: utf-8 -*-
"""Fast VoxForge smoke check.

This script verifies project wiring without loading XTTS weights and without
generating audio. It is intentionally limited to file, import, tool, and local
profile checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = REPORTS_DIR / "smoke_check_report.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_FILES = [
    "app/gradio_xtts_demo.py",
    "scripts/voice_profile_utils.py",
    "scripts/audio_quality_utils.py",
    "scripts/audio_preprocessing_utils.py",
    "scripts/list_voice_profiles.py",
    "scripts/create_voice_profile.py",
    "run_gradio_demo.ps1",
    "run_first_xtts_test.ps1",
    "run_create_voice_profile.ps1",
    "run_list_voice_profiles.ps1",
    "run_audio_quality_report.ps1",
    "run_compare_reference_quality.ps1",
    "run_smoke_check.ps1",
    ".gitignore",
]

REQUIRED_DIRECTORIES = [
    "samples",
    "outputs",
    "profiles",
]

REQUIRED_GITKEEP_FILES = [
    "samples/.gitkeep",
    "outputs/.gitkeep",
    "profiles/.gitkeep",
]

REQUIRED_GITIGNORE_RULES = [
    ".venv/",
    "samples/*",
    "outputs/*",
    "profiles/*",
    "!samples/.gitkeep",
    "!outputs/.gitkeep",
    "!profiles/.gitkeep",
]

IMPORT_CHECKS = [
    ("torch", "torch", None),
    ("gradio", "gradio", None),
    ("from TTS.api import TTS", "TTS.api", "TTS"),
    ("scripts.audio_quality_utils", "scripts.audio_quality_utils", None),
    ("scripts.audio_preprocessing_utils", "scripts.audio_preprocessing_utils", None),
    ("scripts.voice_profile_utils", "scripts.voice_profile_utils", None),
]

PROFILE_JSON_NAME = "profile.json"
ORIGINAL_REFERENCE_NAME = "original_reference.wav"
PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"


class SmokeReport:
    """Collects terminal checks and JSON report data."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def add_ok(self, name: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": "ok", "detail": detail})

    def add_warning(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": "warning", "detail": detail})
        self.warnings.append(f"{name}: {detail}")

    def add_error(self, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": "error", "detail": detail})
        self.errors.append(f"{name}: {detail}")

    def result_text(self) -> str:
        if self.errors:
            return "SMOKE CHECK FAILED"
        if self.warnings:
            return "SMOKE CHECK PASSED WITH WARNINGS"
        return "SMOKE CHECK PASSED"

    def exit_code(self) -> int:
        return 1 if self.errors else 0


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def check_required_paths(report: SmokeReport) -> None:
    for relative_path in REQUIRED_FILES:
        path = project_path(relative_path)
        if path.is_file():
            report.add_ok(f"file:{relative_path}", "found")
        else:
            report.add_error(f"file:{relative_path}", "missing")

    for relative_path in REQUIRED_DIRECTORIES:
        path = project_path(relative_path)
        if path.is_dir():
            report.add_ok(f"dir:{relative_path}", "found")
        else:
            report.add_error(f"dir:{relative_path}", "missing")

    for relative_path in REQUIRED_GITKEEP_FILES:
        path = project_path(relative_path)
        if path.is_file():
            report.add_ok(f"gitkeep:{relative_path}", "found")
        else:
            report.add_error(f"gitkeep:{relative_path}", "missing")


def check_gitignore_rules(report: SmokeReport) -> None:
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if not gitignore_path.is_file():
        report.add_error(".gitignore rules", ".gitignore is missing")
        return

    lines = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    for rule in REQUIRED_GITIGNORE_RULES:
        if rule in lines:
            report.add_ok(f".gitignore:{rule}", "found")
        else:
            report.add_error(f".gitignore:{rule}", "missing")


def import_module_check(name: str, module_name: str, attribute_name: str | None) -> Any:
    module = importlib.import_module(module_name)
    if attribute_name is not None:
        return getattr(module, attribute_name)
    return module


def check_imports(report: SmokeReport) -> dict[str, Any]:
    imported: dict[str, Any] = {}

    for display_name, module_name, attribute_name in IMPORT_CHECKS:
        try:
            imported_value = import_module_check(display_name, module_name, attribute_name)
        except Exception as exc:
            report.add_error(f"import:{display_name}", f"{type(exc).__name__}: {exc}")
            continue

        imported[module_name] = imported_value
        report.add_ok(f"import:{display_name}", "ok")

    return imported


def collect_torch_info(report: SmokeReport) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_available": None,
        "cuda_device_name": None,
    }

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        report.add_error("torch info", f"{type(exc).__name__}: {exc}")
        return info

    info["available"] = True
    info["version"] = getattr(torch, "__version__", "unknown")

    try:
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        if cuda_available:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report.add_warning("torch cuda info", f"{type(exc).__name__}: {exc}")

    detail = (
        f"version={info['version']}, "
        f"cuda_available={info['cuda_available']}, "
        f"cuda_device_name={info['cuda_device_name'] or 'none'}"
    )
    report.add_ok("torch runtime", detail)
    return info


def run_version_command(tool_name: str) -> dict[str, Any]:
    tool_path = shutil.which(tool_name)
    result: dict[str, Any] = {
        "tool": tool_name,
        "path": tool_path,
        "available": tool_path is not None,
        "returncode": None,
        "first_line": None,
        "error": None,
    }

    if tool_path is None:
        result["error"] = f"{tool_name} not found in PATH"
        return result

    try:
        completed = subprocess.run(
            [tool_path, "-version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["returncode"] = completed.returncode
    output = (completed.stdout or completed.stderr).strip()
    result["first_line"] = output.splitlines()[0] if output else ""
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip() or completed.stdout.strip()

    return result


def check_external_tools(report: SmokeReport) -> dict[str, Any]:
    tools: dict[str, Any] = {}

    for tool_name in ("ffmpeg", "ffprobe"):
        tool_info = run_version_command(tool_name)
        tools[tool_name] = tool_info

        if tool_info["available"] and tool_info["returncode"] == 0:
            report.add_ok(
                f"tool:{tool_name}",
                f"{tool_info['path']} | {tool_info['first_line']}",
            )
        else:
            report.add_warning(
                f"tool:{tool_name}",
                str(tool_info.get("error") or "version command failed"),
            )

    return tools


def quality_summary(report_data: Any) -> str:
    if not isinstance(report_data, dict):
        return "UNKNOWN"

    quality = report_data.get("quality") or "UNKNOWN"
    duration = report_data.get("duration_seconds")
    sample_rate = report_data.get("sample_rate")

    details = []
    if duration is not None:
        details.append(f"duration={duration}s")
    if sample_rate is not None:
        details.append(f"sample_rate={sample_rate}Hz")

    if details:
        return f"{quality} ({', '.join(details)})"
    return str(quality)


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except OSError as exc:
        return None, str(exc)

    if not isinstance(data, dict):
        return None, "profile.json is not a JSON object"
    return data, None


def collect_profiles(report: SmokeReport) -> dict[str, Any]:
    profiles_dir = PROJECT_ROOT / "profiles"
    valid_profiles: list[dict[str, Any]] = []
    broken_profiles: list[dict[str, Any]] = []

    if not profiles_dir.is_dir():
        return {
            "valid_profile_count": 0,
            "broken_profile_count": 0,
            "profiles": valid_profiles,
            "broken_profiles": broken_profiles,
        }

    for profile_dir in sorted(profiles_dir.iterdir(), key=lambda item: item.name.lower()):
        if not profile_dir.is_dir():
            continue

        profile_json = profile_dir / PROFILE_JSON_NAME
        original_reference = profile_dir / ORIGINAL_REFERENCE_NAME
        preprocessed_reference = profile_dir / PREPROCESSED_REFERENCE_NAME
        issues: list[str] = []

        if not profile_json.is_file():
            issues.append(f"{PROFILE_JSON_NAME} missing")
            metadata = None
        else:
            metadata, json_error = read_json_file(profile_json)
            if json_error:
                issues.append(json_error)

        if not original_reference.is_file():
            issues.append(f"{ORIGINAL_REFERENCE_NAME} missing")
        if not preprocessed_reference.is_file():
            issues.append(f"{PREPROCESSED_REFERENCE_NAME} missing")

        base_record = {
            "profile_slug": profile_dir.name,
            "profile_dir": str(profile_dir),
            "profile_json_exists": profile_json.is_file(),
            "original_reference_exists": original_reference.is_file(),
            "preprocessed_reference_exists": preprocessed_reference.is_file(),
        }

        if issues or metadata is None:
            broken_profiles.append({**base_record, "issues": issues})
            continue

        valid_profiles.append(
            {
                **base_record,
                "profile_name": metadata.get("profile_name") or profile_dir.name,
                "created_at": metadata.get("created_at"),
                "selected_preprocessing_variant": metadata.get(
                    "selected_preprocessing_variant"
                ),
                "preprocessing_warning": metadata.get("preprocessing_warning"),
                "original_quality_summary": quality_summary(
                    metadata.get("original_quality")
                ),
                "preprocessed_quality_summary": quality_summary(
                    metadata.get("preprocessed_quality")
                ),
            }
        )

    if broken_profiles:
        report.add_warning(
            "profiles",
            f"{len(broken_profiles)} broken or incomplete profile(s) found",
        )
    else:
        report.add_ok("profiles", "no broken profiles found")

    return {
        "valid_profile_count": len(valid_profiles),
        "broken_profile_count": len(broken_profiles),
        "profiles": valid_profiles,
        "broken_profiles": broken_profiles,
    }


def ensure_reports_dir(report: SmokeReport) -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.add_error("outputs/reports", f"could not create reports directory: {exc}")
        return

    report.add_ok("outputs/reports", f"ready: {REPORTS_DIR}")


def build_json_report(
    report: SmokeReport,
    torch_info: dict[str, Any],
    tools: dict[str, Any],
    profiles: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checks": report.checks,
        "warnings": report.warnings,
        "errors": report.errors,
        "summary": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "result": report.result_text(),
            "exit_code": report.exit_code(),
            "warning_count": len(report.warnings),
            "error_count": len(report.errors),
            "torch": torch_info,
            "tools": tools,
            "profiles": profiles,
        },
    }


def write_json_report(report_data: dict[str, Any], report: SmokeReport) -> None:
    try:
        REPORT_PATH.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        report.add_error("json report", f"could not write {REPORT_PATH}: {exc}")


def print_check_table(report: SmokeReport) -> None:
    print("")
    print("Checks")
    print("------")
    for check in report.checks:
        status = check["status"].upper()
        detail = check.get("detail") or ""
        print(f"[{status}] {check['name']} - {detail}")


def print_profile_summary(profiles: dict[str, Any]) -> None:
    print("")
    print("Profiles")
    print("--------")
    print(f"Valid profiles: {profiles['valid_profile_count']}")
    print(f"Broken profiles: {profiles['broken_profile_count']}")

    for profile in profiles["profiles"]:
        print(
            "- "
            f"{profile['profile_slug']} | "
            f"preprocessed={profile['preprocessed_quality_summary']} | "
            f"original={profile['original_quality_summary']}"
        )

    for profile in profiles["broken_profiles"]:
        issues = "; ".join(profile.get("issues", [])) or "unknown issue"
        print(f"- BROKEN {profile['profile_slug']} | {issues}")


def main() -> int:
    report = SmokeReport()

    print("VoxForge Smoke Check")
    print("====================")
    print(f"Project root: {PROJECT_ROOT}")
    print("This check does not load XTTS weights and does not generate audio.")

    check_required_paths(report)
    check_gitignore_rules(report)
    check_imports(report)
    torch_info = collect_torch_info(report)
    tools = check_external_tools(report)
    profiles = collect_profiles(report)
    ensure_reports_dir(report)

    report_data = build_json_report(report, torch_info, tools, profiles)
    write_json_report(report_data, report)

    if report.errors and report_data["summary"]["error_count"] != len(report.errors):
        report_data = build_json_report(report, torch_info, tools, profiles)
        write_json_report(report_data, report)

    print_check_table(report)
    print_profile_summary(profiles)
    print("")
    print(f"JSON report: {REPORT_PATH}")
    print(report.result_text())

    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
