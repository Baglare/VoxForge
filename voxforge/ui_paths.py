"""Gradio dosya izinleri (junction güvenli, dar kapsam).

Gradio bir dosyayı sunmadan önce *çözülmüş* (symlink/junction takip edilmiş) yolun izinli bir
klasörde olup olmadığına bakar. ``profiles/`` ve ``outputs/`` birer junction olabileceğinden
(``C:/AvenoxData`` altına çözülür) izinli klasörler de çözülmüş yollarıyla verilmelidir.
Yalnız uygulamanın kendi ses klasörleri izinlidir; proje kökü, veri kümeleri ve eğitim
çıktıları değildir.
"""

from __future__ import annotations

from pathlib import Path


def gradio_allowed_paths(*extra: Path | str) -> list[str]:
    """Resolved, de-duplicated folders Gradio may serve audio from (created when missing)."""
    from scripts.voice_profile_utils import PROFILES_DIR
    from voxforge.studio import studio_root
    from voxforge.voices.library import DEFAULT_TRASH

    repo_outputs = Path(__file__).resolve().parents[1] / "outputs"
    candidates: list[Path | str] = [
        PROFILES_DIR,
        studio_root(),
        repo_outputs / "playground",
        repo_outputs / "gradio_outputs",
        repo_outputs / "preprocessed_references",
        repo_outputs / "reports",
        DEFAULT_TRASH,
        *extra,
    ]
    result: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            resolved = str(path.resolve())
        except OSError:
            continue
        if resolved not in result:
            result.append(resolved)
    return result
