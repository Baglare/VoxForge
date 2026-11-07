"""VoxCPM2 advanced voice controls, presets and narration pacing (VF-016).

UI independent; used by the Playground service and the scene narration module, and
meant to move into Studio unchanged.

What VoxCPM2 really supports (voxcpm 2.0.3 + model card, checked in the installed code):
- ``cfg_value`` and ``inference_timesteps`` (generate arguments); seed via torch RNG;
- *controllable cloning*: a natural-language instruction rendered as ``(instruction)text``
  steers emotion / pace / expression while the reference keeps the timbre. The model card
  warns results vary between runs. It cannot be combined with Ultimate Cloning
  (prompt_text continuation); the library CLI rejects that pair as well.
- There is no numeric speed / emotion parameter. "Speed" here = instruction + pauses.

Narration pacing (measured in docs/VF016_PROSODY_FINDINGS.md): long text is generated
one sentence per call and joined with trimmed edges and explicit pauses, instead of
packing ~250 characters per call. Packed chunks left 0.02-0.4 s between sentences, which
is what made long narration sound rushed; per-sentence calls also kept the speaker
similarity higher. Short text (one sentence / one packed chunk) is generated exactly
as before.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

CFG_RANGE = (1.0, 3.0)
STEPS_RANGE = (4, 20)
LOW_STEPS_WARNING = 8  # VF-015: 6 adımda telaffuz bozuldu
PAUSE_RANGE = (0.0, 2.0)
STYLE_MAX_CHARS = 160
PACINGS = {"standard": "Standart (cümleler ~250 karaktere paketlenir; önceki davranış)", "narration": "Anlatım (cümle cümle üretim + doğal duraklama)"}
SUBCHUNK_PAUSE = 0.15  # tek cümle max_chars'ı aşıp virgülden bölündüğünde
PRESET_NAME = re.compile(r"^[\wçğıöşüÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ .\-]{0,39}$")


class ControlError(ValueError):
    """User-facing invalid control value."""


@dataclass(frozen=True)
class VoxControls:
    cfg: float = 2.0
    steps: int = 10
    style: str = ""
    pacing: str = "standard"
    sentence_pause: float = 0.45
    paragraph_pause: float = 0.85

    def validated(self) -> "VoxControls":
        cfg = round(float(self.cfg), 2)
        steps = int(self.steps)
        if not CFG_RANGE[0] <= cfg <= CFG_RANGE[1]:
            raise ControlError(f"CFG {CFG_RANGE[0]}-{CFG_RANGE[1]} aralığında olmalı ({cfg}).")
        if not STEPS_RANGE[0] <= steps <= STEPS_RANGE[1]:
            raise ControlError(f"Difüzyon adımı {STEPS_RANGE[0]}-{STEPS_RANGE[1]} aralığında olmalı ({steps}).")
        if self.pacing not in PACINGS:
            raise ControlError(f"Bilinmeyen anlatım ritmi: {self.pacing}")
        for name in ("sentence_pause", "paragraph_pause"):
            value = float(getattr(self, name))
            if not PAUSE_RANGE[0] <= value <= PAUSE_RANGE[1]:
                raise ControlError(f"Duraklama {PAUSE_RANGE[0]}-{PAUSE_RANGE[1]} sn olmalı ({value}).")
        style = " ".join((self.style or "").replace("(", " ").replace(")", " ").split())
        if len(style) > STYLE_MAX_CHARS:
            raise ControlError(f"Stil yönergesi en fazla {STYLE_MAX_CHARS} karakter olabilir.")
        return replace(
            self, cfg=cfg, steps=steps, style=style,
            sentence_pause=round(float(self.sentence_pause), 2), paragraph_pause=round(float(self.paragraph_pause), 2),
        )

    def engine_settings(self) -> dict[str, Any]:
        """Only what differs from the VF-Q1 defaults is sent (so defaults stay byte-identical)."""
        settings: dict[str, Any] = {}
        if self.cfg != 2.0:
            settings["cfg_value"] = self.cfg
        if self.steps != 10:
            settings["inference_timesteps"] = self.steps
        return settings

    def warnings(self, ultimate: bool = False) -> list[str]:
        notes = []
        if self.steps < LOW_STEPS_WARNING:
            notes.append(f"{self.steps} difüzyon adımı: VF-015 pilotunda 6 adımda telaffuz bozuldu.")
        if ultimate and self.style:
            notes.append("Stil yönergesi Ultimate Cloning ile kullanılamaz.")
        return notes

    def label(self) -> str:
        parts = [f"cfg {self.cfg:g}", f"{self.steps} adım"]
        if self.style:
            parts.append(f"stil «{self.style}»")
        if self.pacing == "narration":
            parts.append(f"anlatım ritmi ({self.sentence_pause:g}/{self.paragraph_pause:g} sn)")
        return " · ".join(parts)

    def is_default(self) -> bool:
        return self.validated() == VoxControls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VoxControls":
        known = {name: value for name, value in (data or {}).items() if name in cls.__dataclass_fields__}
        return cls(**known).validated()


# ---- compatibility with Ultimate Cloning -------------------------------------------------------
ULTIMATE_COMPATIBILITY = {
    "cfg": "uyumlu",
    "steps": "uyumlu",
    "seed": "uyumlu",
    "pacing": "uyumlu (her cümle referans + transkript devamıyla üretilir)",
    "style": "UYUMSUZ: Ultimate prompt devamı yapar; stil yönergesi reddedilir",
}


# ---- presets ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    controls: VoxControls
    evidence: str  # ne ölçüldü / ne denenmedi (başarılı gibi gösterme)
    builtin: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "controls": asdict(self.controls), "evidence": self.evidence,
                "builtin": self.builtin, "tags": list(self.tags)}


# Evidence: docs/VF016_PROSODY_FINDINGS.md (mimo profile, seed 1234 [+77], faster-whisper word
# timestamps, ECAPA similarity to the reference). In Turkish every style instruction tested
# acted mainly as "slower + longer pauses"; none produced a measurable speed-up and
# "fast/energetic" instructions moved speaker similarity a lot between seeds, so the action
# preset carries no instruction. Emotional effect itself cannot be measured automatically.
BUILTIN_PRESETS: dict[str, Preset] = {
    preset.key: preset
    for preset in (
        Preset(
            "normal", "Normal (varsayılan)", VoxControls(),
            "VF-Q1/VF-015 varsayılanı (cfg 2.0 · 10 adım). Kısa metin çıktısı VF-016'da bayt bayt aynı kaldı. "
            "Uzun anlatımda ~7,0 hece/sn ve cümle arası 0,02-0,4 sn: acele hissinin kaynağı.",
        ),
        Preset(
            "sakin", "Sakin anlatım",
            VoxControls(style="sakin, yumuşak ve dengeli bir anlatım", pacing="narration", sentence_pause=0.5),
            "Ölçüldü: kısa metinde 7,5 → 6,1 hece/sn, ses benzerliği normal ile aynı (0,737); CER 0.",
            tags=("calm",),
        ),
        Preset(
            "yavas-anlatici", "Yavaş anlatıcı",
            VoxControls(style="yavaş, sakin bir anlatım", pacing="narration", sentence_pause=0.55, paragraph_pause=0.95),
            "Ölçüldü (uzun anlatım için önerilen): 6,93/7,05 → 6,38/6,22 hece/sn, cümle arası duraklama toplamı "
            "~2 → ~3,6 sn, ilk iki cümlenin acelesi kalktı (oran 1,09 → 0,97); benzerlik 0,824/0,852 → 0,832/0,845; CER ≤0,003.",
            tags=("calm", "slow"),
        ),
        Preset(
            "dramatik", "Dramatik",
            VoxControls(style="dramatik, etkileyici ve vurgulu bir anlatım", pacing="narration", sentence_pause=0.6),
            "Kısmen ölçüldü: tempo 7,5 → 6,1 hece/sn, benzerlik ~aynı (0,728). Dramatik vurgu otomatik ölçülemez; dinleyerek doğrulayın.",
            tags=("dramatic",),
        ),
        Preset(
            "gerilim", "Gerilim / gizem",
            VoxControls(style="gergin, gizemli ve alçak sesle bir anlatım", pacing="narration", sentence_pause=0.6),
            "Kısmen ölçüldü: tempo 7,5 → 6,1 hece/sn, benzerlik ~aynı (0,728). Gerilim etkisi dinleyerek doğrulanmalı.",
            tags=("tension",),
        ),
        Preset(
            "aksiyon", "Savaş / aksiyon",
            VoxControls(pacing="narration", sentence_pause=0.3),
            "Stil yönergesi YOK, yalnız kısa duraklama. 'hızlı, enerjik' türü yönergeler ölçümde hızlandırmadı "
            "(hep yavaşlattı) ve ses benzerliğini tohuma göre 0,40-0,69 arasında oynattı.",
            tags=("action",),
        ),
        Preset(
            "duygusal", "Duygusal",
            VoxControls(style="yumuşak, duygulu bir anlatım", pacing="narration", sentence_pause=0.55),
            "Deneysel: benzerlik yönergesizle yakın (0,70/0,68 vs 0,70/0,73), tempo tohuma göre değişti. "
            "Duygu etkisi dinleyerek doğrulanmalı; bazı üretimlerde kimlik hafif kayabilir (1-3 deneme önerilir).",
            tags=("emotional",),
        ),
        Preset(
            "diyalog", "Diyalog (doğal konuşma)",
            VoxControls(pacing="narration", sentence_pause=0.35),
            "Stil yönergesi yok: cümle cümle üretim uzun metinde benzerliği artırdı (0,824/0,852 → 0,848/0,861), tempo aynı.",
            tags=("dialogue",),
        ),
    )
}


class PresetStore:
    """Built-in presets + the user's own, saved as JSON next to Playground outputs (git ignored)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load_user(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data.get("presets", {}) if isinstance(data, dict) else {}

    def _write_user(self, presets: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".partial")
        temporary.write_text(json.dumps({"version": 1, "presets": presets}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def all(self) -> dict[str, Preset]:
        presets = dict(BUILTIN_PRESETS)
        for key, item in self._load_user().items():
            try:
                controls = VoxControls.from_dict(item.get("controls"))
            except (ControlError, TypeError, ValueError):
                continue
            presets[key] = Preset(key, item.get("name", key), controls, item.get("evidence", "Kullanıcı preset'i"), builtin=False)
        return presets

    def get(self, key: str) -> Preset:
        presets = self.all()
        if key not in presets:
            raise ControlError(f"Bilinmeyen preset: {key}")
        return presets[key]

    @staticmethod
    def key_for(name: str) -> str:
        return "user-" + re.sub(r"[^a-z0-9]+", "-", name.lower().translate(str.maketrans("çğıöşü", "cgiosu"))).strip("-")

    def save(self, name: str, controls: VoxControls) -> Preset:
        name = " ".join((name or "").split())
        if not PRESET_NAME.match(name):
            raise ControlError("Preset adı 1-40 karakter olmalı (harf, rakam, boşluk, . - _).")
        controls = controls.validated()
        key = self.key_for(name)
        presets = self._load_user()
        presets[key] = {"name": name, "controls": asdict(controls), "evidence": "Kullanıcı preset'i (ölçülmedi)"}
        self._write_user(presets)
        return Preset(key, name, controls, presets[key]["evidence"], builtin=False)

    def delete(self, key: str) -> None:
        if key in BUILTIN_PRESETS:
            raise ControlError("Hazır preset'ler silinemez.")
        presets = self._load_user()
        if presets.pop(key, None) is None:
            raise ControlError(f"Preset bulunamadı: {key}")
        self._write_user(presets)


# ---- chunk plan ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkPlan:
    spoken: str
    chunks: list[str]
    pauses: list[float] | None  # None = legacy join (0.2 s, untrimmed)
    trim: bool


def plan_chunks(text: str, controls: VoxControls, max_chars: int, normalize: bool) -> ChunkPlan:
    """Synthesis chunks for ``text``.

    standard: the original Playground behaviour (sentences packed to ``max_chars``).
    narration: one VF-004 sentence segment per chunk; pauses come from the paragraph
    structure of the *original* text (blank line = paragraph); a sentence that is still
    longer than ``max_chars`` after normalisation is split at commas with a short pause.
    """
    from scripts.text_chunking_utils import split_text_for_tts

    if controls.pacing == "standard":
        from voxforge.playground.service import split_for_engine

        spoken, chunks = split_for_engine(text, max_chars, normalize)
        return ChunkPlan(spoken, chunks, None, False)

    from voxforge.text import prepare_segments, segment_text

    if normalize:
        segments = [(item.segment.paragraph, item.spoken) for item in prepare_segments(text, max_chars=max_chars)]
    else:
        segments = [(segment.paragraph, segment.text) for segment in segment_text(text, max_chars=max_chars)]
    chunks: list[str] = []
    pauses: list[float] = []
    previous_paragraph = None
    for paragraph, sentence in segments:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        pieces = [piece for piece in split_text_for_tts(sentence, max_chars=max_chars) if piece.strip()]
        for index, piece in enumerate(pieces):
            if chunks:
                if index:
                    pauses.append(SUBCHUNK_PAUSE)
                elif paragraph != previous_paragraph:
                    pauses.append(controls.paragraph_pause)
                else:
                    pauses.append(controls.sentence_pause)
            chunks.append(piece)
        previous_paragraph = paragraph
    spoken = " ".join(chunks)
    return ChunkPlan(spoken, chunks, pauses, True)
