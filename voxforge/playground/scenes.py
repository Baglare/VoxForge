"""Scene-based narration (VF-016): one text, several scenes, one voice, one WAV.

UI independent and Studio-portable: a ``SceneProject`` is plain JSON (scenes with text,
kind and preset), generation goes through ``PlaygroundService.generate`` (so every scene
gets the normal record with model, settings, timing and VRAM), and the merge is
``audio_join.join_wavs``.

Voice identity: every scene of a project uses the same profile reference, the same engine
variant and the same seed; only the preset (style instruction, pauses, cfg/steps) changes
per scene. Scene kinds are interface shortcuts that map to presets; they do not claim an
emotion control the model does not have (see ``prosody.BUILTIN_PRESETS`` evidence).

Storage: ``outputs/playground/scenes/<slug>.json`` and audio under
``outputs/playground/scenes/<slug>/`` (git ignored, like all Playground outputs).

Validity: each generated scene stores a *fingerprint* of everything that shaped its audio
(text, profile + reference file identity, variant and its fixed settings, seed,
normalisation, the preset's real control values, Ultimate transcript, device). A scene is
ready only while the fingerprint of the current settings matches; otherwise it is
regenerated and merging is refused. Old project files (no fingerprint) still load; their
scenes count as "needs regeneration" because their settings cannot be proven.

New projects never overwrite an existing project file: ``SceneStore.create`` reserves the
slug atomically and picks ``<slug>-2``, ``-3`` ... when the name is taken.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voxforge.playground.audio_join import join_wavs
from voxforge.playground.prosody import ControlError, VoxControls

SCENE_KINDS: dict[str, tuple[str, str]] = {  # kind -> (label, default preset key)
    "calm": ("Sakin anlatım", "sakin"),
    "tension": ("Gerilim ve gizem", "gerilim"),
    "action": ("Savaş / aksiyon", "aksiyon"),
    "emotional": ("Duygusal sahne", "duygusal"),
    "dialogue": ("Diyalog", "diyalog"),
    "dramatic": ("Dramatik konuşma", "dramatik"),
    "normal": ("Normal", "normal"),
}
MAX_SCENES = 40
MAX_SCENE_CHARS = 2000
SCENE_GAP_SECONDS = 0.8
PROJECT_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
FINGERPRINT_VERSION = 1
MAX_SLUG_SUFFIX = 999
SCENE_MARKER = re.compile(r"^\s*(?:#{1,6}\s+(?P<title>.+?)|-{3,}|\*{3,}|\[sahne(?::\s*(?P<title2>[^\]]*))?\])\s*$", re.IGNORECASE)

# Keyword cues for the optional suggestion (lower-case stems; Turkish suffixes follow).
KIND_CUES: dict[str, tuple[str, ...]] = {
    "action": ("savaş", "kılıç", "saldır", "koş", "patla", "vur", "çarpış", "ok yağ", "hücum", "kaç", "düşman", "kalkan", "mızrak", "ateş"),
    "tension": ("karanlık", "gölge", "fısılt", "ayak ses", "sessizlik", "korku", "ürper", "gizem", "kimse", "rivayet", "mühür", "titre", "yalnız"),
    "emotional": ("gözyaş", "ağla", "özle", "sevgi", "veda", "kalbi", "hüzün", "anne", "kayıp", "affet", "sarıl"),
    "dramatic": ("asla", "yemin", "kader", "son kez", "bütün dünya", "intikam", "!"),
}
DIALOGUE_LINE = re.compile(r'^\s*(?:[—–-]\s|["“«])')


def slugify(name: str) -> str:
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").translate(table).lower()).strip("-")[:48]
    if not slug or not PROJECT_SLUG.match(slug):
        raise ControlError("Proje adı en az bir harf veya rakam içermeli.")
    return slug


def suggest_kind(text: str) -> str:
    """Cheap, local, editable suggestion (no LLM, no service): dialogue lines first, then cue counts."""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and sum(1 for line in lines if DIALOGUE_LINE.match(line)) >= max(1, len(lines) // 2):
        return "dialogue"
    lowered = text.lower()
    scores = {kind: sum(lowered.count(cue) for cue in cues) for kind, cues in KIND_CUES.items()}
    scores["dramatic"] = scores["dramatic"] + max(0, lowered.count("!") - 1)
    kind, score = max(scores.items(), key=lambda item: item[1])
    return kind if score >= 2 else "calm"


@dataclass
class Scene:
    id: str
    title: str
    text: str
    kind: str = "calm"
    preset: str = "sakin"
    record_id: str | None = None
    wav: str | None = None
    audio_seconds: float | None = None
    generated_text: str | None = None  # bilgi amaçlı; geçerlilik ``fingerprint`` ile belirlenir
    generated_preset: str | None = None
    fingerprint: str | None = None  # üretimi belirleyen ayarların özeti (yoksa eski kayıt: yeniden üretilir)
    device: str | None = None


@dataclass
class SceneProject:
    name: str
    slug: str
    profile: str | None = None
    variant: str = "voxcpm2"
    seed: int = 1234
    normalize: bool = True
    scene_gap: float = SCENE_GAP_SECONDS
    scenes: list[Scene] = field(default_factory=list)
    merged_wav: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneProject":
        scenes = [Scene(**{k: v for k, v in item.items() if k in Scene.__dataclass_fields__}) for item in data.get("scenes", [])]
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "scenes"}
        return cls(**fields, scenes=scenes)


def split_scenes(text: str, suggest: bool = True) -> list[Scene]:
    """Scenes from markers (``## Başlık``, ``---``, ``[sahne: Başlık]``) or, without markers, paragraphs."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    blocks: list[tuple[str | None, list[str]]] = []
    has_marker = any(SCENE_MARKER.match(line) for line in text.split("\n"))
    if has_marker:
        current: tuple[str | None, list[str]] = (None, [])
        for line in text.split("\n"):
            match = SCENE_MARKER.match(line)
            if match:
                if any(part.strip() for part in current[1]):
                    blocks.append(current)
                current = ((match.group("title") or match.group("title2") or "").strip() or None, [])
            else:
                current[1].append(line)
        if any(part.strip() for part in current[1]):
            blocks.append(current)
    else:
        blocks = [(None, [paragraph]) for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if len(blocks) > MAX_SCENES:
        raise ControlError(f"En fazla {MAX_SCENES} sahne desteklenir ({len(blocks)}).")
    scenes = []
    for index, (title, lines) in enumerate(blocks, start=1):
        body = "\n".join(lines).strip()
        if len(body) > MAX_SCENE_CHARS:
            raise ControlError(f"Sahne {index} en fazla {MAX_SCENE_CHARS} karakter olabilir ({len(body)}).")
        kind = suggest_kind(body) if suggest else "calm"
        scenes.append(Scene(id=f"s{index:02d}", title=title or f"Sahne {index}", text=body, kind=kind, preset=SCENE_KINDS[kind][1]))
    return scenes


class SceneStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, slug: str) -> Path:
        if not PROJECT_SLUG.match(slug or ""):
            raise ControlError(f"Geçersiz proje adı: {slug!r}")
        return self.root / f"{slug}.json"

    def audio_dir(self, slug: str) -> Path:
        return self.path(slug).with_suffix("")

    def list(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json") if PROJECT_SLUG.match(path.stem))

    def load(self, slug: str) -> SceneProject:
        path = self.path(slug)
        if not path.is_file():
            raise ControlError(f"Sahne projesi bulunamadı: {slug}")
        return SceneProject.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def create(self, project: SceneProject) -> SceneProject:
        """Save a NEW project without touching existing ones.

        The slug file is reserved with an exclusive create, so neither an existing project
        nor a concurrent creation is overwritten; a taken name gets ``-2``, ``-3`` ...
        (``project.slug`` and ``project.name`` are updated to the name actually used).
        """
        base = project.slug
        self.root.mkdir(parents=True, exist_ok=True)
        for number in range(1, MAX_SLUG_SUFFIX + 1):
            suffix = "" if number == 1 else f"-{number}"
            slug = base[: 48 - len(suffix)].rstrip("-") + suffix
            if self.audio_dir(slug).exists():
                continue  # sesleri kalmış eski bir proje klasörünü de sahiplenme
            try:
                with self.path(slug).open("x", encoding="utf-8"):
                    pass
            except FileExistsError:
                continue
            if number > 1:
                project.name = f"{project.name.strip()}-{number}"
            project.slug = slug
            return self.save(project)
        raise ControlError(f"'{base}' adıyla çok fazla proje var; başka bir ad seçin.")

    def save(self, project: SceneProject) -> SceneProject:
        project.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        path = self.path(project.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return project


class SceneNarrator:
    """Generates scenes through a PlaygroundService and merges them."""

    def __init__(self, service: Any, store: SceneStore | None = None) -> None:
        self.service = service
        self.store = store or SceneStore(Path(service.root) / "scenes")

    def _controls(self, preset_key: str) -> VoxControls:
        return self.service.presets.get(preset_key).controls

    def _reference_identity(self, profile: str | None) -> dict[str, Any]:
        """Which reference file the profile resolves to now (a re-recorded profile changes it)."""
        if not profile:
            return {"path": None}
        from voxforge.playground.service import resolve_profile_reference

        try:
            path = Path(resolve_profile_reference(profile))
            stat = path.stat()
        except Exception as exc:  # noqa: BLE001 - kayıp profil: parmak izi değişir, sahne eski sayılır
            return {"path": None, "error": type(exc).__name__}
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def fingerprint(self, project: SceneProject, scene: Scene, device: str = "cuda") -> str:
        """Hash of every input that shapes this scene's audio (see module docstring)."""
        from voxforge.playground.service import VARIANTS

        try:
            controls: Any = asdict(self._controls(scene.preset))
        except ControlError:
            controls = {"missing_preset": scene.preset}
        variant = VARIANTS.get(project.variant)
        transcript = None
        if variant is not None and variant.ultimate:
            transcript = self.service.get_transcript(project.profile).strip()
        payload = {
            "v": FINGERPRINT_VERSION,
            "text": scene.text,
            "profile": project.profile,
            "reference": self._reference_identity(project.profile),
            "variant": project.variant,
            "variant_settings": dict(variant.settings) if variant else None,
            "max_chars": variant.max_chars if variant else None,
            "seed": int(project.seed),
            "normalize": bool(project.normalize),
            "preset": scene.preset,
            "controls": controls,
            "transcript": transcript,
            "device": device,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def scene_state(self, project: SceneProject, scene: Scene, device: str = "cuda") -> str:
        """``missing`` (no audio), ``stale`` (settings changed / legacy record) or ``ready``."""
        if not scene.wav or not Path(scene.wav).is_file():
            return "missing"
        if not scene.fingerprint or scene.fingerprint != self.fingerprint(project, scene, device):
            return "stale"
        return "ready"

    def is_stale(self, project: SceneProject, scene: Scene, device: str = "cuda") -> bool:
        return self.scene_state(project, scene, device) != "ready"

    def generate_scene(self, project: SceneProject, scene_id: str, device: str = "cuda") -> Scene:
        scene = next((item for item in project.scenes if item.id == scene_id), None)
        if scene is None:
            raise ControlError(f"Sahne bulunamadı: {scene_id}")
        if not project.profile:
            raise ControlError("Sahne anlatımı için bir ses profili seçin.")
        if scene.kind not in SCENE_KINDS:
            raise ControlError(f"Bilinmeyen sahne türü: {scene.kind}")
        fingerprint = self.fingerprint(project, scene, device)  # üretimden önce: kullanılan girdiler
        record = self.service.generate(
            scene.text,
            project.variant,
            project.profile,
            seed=project.seed,  # tüm sahnelerde aynı tohum + aynı referans = aynı ses kimliği
            normalize=project.normalize,
            device=device,
            controls=self._controls(scene.preset),
            preset_key=scene.preset,
            out_dir=self.store.audio_dir(project.slug),
        )
        old = Path(scene.wav) if scene.wav else None
        scene.record_id, scene.wav = record["id"], record["wav"]
        scene.audio_seconds = record["timing"]["audio_seconds"]
        scene.generated_text, scene.generated_preset = scene.text, scene.preset
        scene.fingerprint, scene.device = fingerprint, device
        if old and old != Path(scene.wav) and old.parent == self.store.audio_dir(project.slug):
            old.unlink(missing_ok=True)  # yalnız bu projenin eski sahne sesi (yeniden üretildi)
        project.merged_wav = None
        self.store.save(project)
        return scene

    def generate_all(self, project: SceneProject, only_stale: bool = True, device: str = "cuda") -> list[str]:
        errors = []
        for scene in project.scenes:
            if only_stale and not self.is_stale(project, scene, device):
                continue
            try:
                self.generate_scene(project, scene.id, device=device)
            except Exception as exc:  # noqa: BLE001 - bir sahne diğerlerini durdurmaz
                errors.append(f"{scene.title}: {exc}")
        return errors

    def merge(self, project: SceneProject, device: str = "cuda") -> dict[str, Any]:
        missing = [scene.title for scene in project.scenes if self.is_stale(project, scene, device)]
        if missing:
            raise ControlError(
                "Önce şu sahneleri üretin (ses yok ya da metin, profil/referans, varyant, tohum, normalizasyon, "
                "preset değerleri veya cihaz değişmiş): " + ", ".join(missing)
            )
        out = self.store.audio_dir(project.slug) / f"{project.slug}_tum_sahneler.wav"
        info = join_wavs(
            [Path(scene.wav) for scene in project.scenes],
            out,
            pauses=project.scene_gap,
            trim=True,
            match_loudness=True,
        )
        project.merged_wav = str(out)
        self.store.save(project)
        return {"wav": str(out), **info}
