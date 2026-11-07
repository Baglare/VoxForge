"""Voice Playground service (UI independent).

Free Turkish text -> one WAV per *variant* (engine + settings) with a JSON record of
model, settings, timing and VRAM. Results live under ``outputs/playground/`` (git
ignored) and are reloaded on start, so comparisons survive a restart.

Nothing here changes engine defaults, Studio projects, benchmark runs or profiles:
profiles are only read, and each variant's settings are explicit overrides on top of
the library defaults used by the VF-Q1 benchmark.

VF-015: generation runs on CUDA unless CPU is chosen explicitly (comparison only); every
record keeps the real runtime (torch build, GPU, dtype), per-chunk timing, VRAM and a short
bottleneck diagnosis. VoxCPM2 Ultimate Cloning (reference + its exact transcript) is an
optional variant; transcripts are kept per profile under ``outputs/playground/transcripts``
(profiles themselves stay read only).
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxforge.playground.workers import REPO_ROOT, EngineManager
from voxforge.runs import utc_now_iso, write_json_atomic

PLAYGROUND_DIR = REPO_ROOT / "outputs" / "playground"
MAX_TEXT_CHARS = 2000
DEFAULT_SEED = 1234  # VF-Q1 ile aynı
HISTORY_LIMIT = 60
MAX_TRANSCRIPT_CHARS = 1000
DEVICE_LABELS = {"cuda": "GPU (CUDA)", "cpu": "CPU (yalnız karşılaştırma)"}
SLUG_PATTERN = r"[A-Za-z0-9_.-]+"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    engine: str
    settings: dict[str, Any] = field(default_factory=dict)
    settings_label: str = "varsayılan"
    max_chars: int = 220
    needs_reference: bool = True
    ultimate: bool = False  # VoxCPM2: referans + birebir transkript (prompt devamı)


VARIANTS: dict[str, Variant] = {
    v.key: v
    for v in (
        Variant("xtts", "XTTS-v2", "xtts", settings_label="varsayılan (Studio ile aynı)"),
        Variant("chatterbox-default", "Chatterbox V3 · varsayılan", "chatterbox-v3", max_chars=250),
        Variant(
            "chatterbox-minp",
            "Chatterbox V3 · min_p=0.02",
            "chatterbox-v3",
            {"min_p": 0.02},
            "min_p=0.02 (kısa ifade tekrarı deneyi)",
            max_chars=250,
        ),
        Variant("voxcpm2", "VoxCPM2", "voxcpm2", settings_label="cfg 2.0 · 10 adım (VF-Q1, varsayılan)", max_chars=250),
        Variant(
            "voxcpm2-fast",
            "VoxCPM2 · hızlı (6 adım)",
            "voxcpm2",
            {"inference_timesteps": 6},
            "cfg 2.0 · 6 difüzyon adımı (hız denemesi)",
            max_chars=250,
        ),
        Variant(
            "voxcpm2-ultimate",
            "VoxCPM2 · Ultimate Cloning",
            "voxcpm2",
            settings_label="cfg 2.0 · 10 adım · referans + transkript",
            max_chars=250,
            ultimate=True,
        ),
        Variant("mock", "Mock (modelsiz test)", "mock", needs_reference=False),
    )
}
ENGINE_LABELS = {"xtts": "XTTS-v2", "chatterbox-v3": "Chatterbox V3", "voxcpm2": "VoxCPM2", "mock": "Mock"}


class PlaygroundError(ValueError):
    """User-facing input problem."""


def split_for_engine(text: str, max_chars: int, normalize: bool) -> tuple[str, list[str]]:
    """Spoken text + synthesis chunks.

    Normalisation is VF-004 (per sentence, same as Studio/benchmark), but chunks are
    packed up to ``max_chars`` like the XTTS demo: a sentence is NOT its own chunk.
    Synthesising every sentence separately added ~0.9 s gaps and a prosody reset
    between sentences (measured on Chatterbox); the benchmark synthesises each item whole.
    """
    from scripts.text_chunking_utils import split_text_for_tts

    if normalize:
        from voxforge.text import prepare_segments

        spoken = " ".join(segment.spoken for segment in prepare_segments(text, max_chars=max_chars))
    else:
        spoken = text
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken, [chunk for chunk in split_text_for_tts(spoken, max_chars=max_chars) if chunk.strip()]


def list_profiles() -> list[dict[str, Any]]:
    """Valid saved voice profiles (read only)."""
    from scripts.voice_profile_utils import list_voice_profiles

    profiles = []
    for item in list_voice_profiles().get("profiles", []):
        reference = item.get("preprocessed_reference")
        if reference and Path(reference).is_file():
            profiles.append({"slug": item["profile_slug"], "name": item.get("profile_name") or item["profile_slug"], "reference": str(reference)})
    return profiles


def resolve_profile_reference(slug: str) -> Path:
    from scripts.voice_profile_utils import get_voice_profile

    return Path(get_voice_profile(slug).preprocessed_reference)


class PlaygroundService:
    def __init__(self, root: Path = PLAYGROUND_DIR, manager: EngineManager | None = None) -> None:
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.transcript_dir = self.root / "transcripts"
        self.manager = manager or EngineManager(self.root / "logs")

    # -- reference transcripts (Ultimate Cloning) ------------------------------------------
    def _transcript_path(self, slug: str) -> Path:
        if not re.fullmatch(SLUG_PATTERN, slug or ""):
            raise PlaygroundError(f"Geçersiz profil adı: {slug!r}")
        return self.transcript_dir / f"{slug}.txt"

    def get_transcript(self, slug: str | None) -> str:
        if not slug:
            return ""
        path = self._transcript_path(slug)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def save_transcript(self, slug: str, text: str) -> None:
        text = (text or "").strip()
        if len(text) > MAX_TRANSCRIPT_CHARS:
            raise PlaygroundError(f"Transkript en fazla {MAX_TRANSCRIPT_CHARS} karakter olabilir.")
        path = self._transcript_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    # -- history ----------------------------------------------------------------------------
    def history(self, limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.audio_dir.glob("*.json"), reverse=True)[: limit * 2]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if Path(record.get("wav", "")).is_file():
                records.append(record)
        return records[:limit]

    def get(self, record_id: str) -> dict[str, Any] | None:
        path = self.audio_dir / f"{record_id}.json"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", record_id) or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- generation -------------------------------------------------------------------------
    def generate(
        self,
        text: str,
        variant_key: str,
        profile_slug: str | None,
        seed: int | None = DEFAULT_SEED,
        normalize: bool = True,
        reference: Path | None = None,
        device: str = "cuda",
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        """Generate one variant; the record is written only for a successful WAV."""
        if variant_key not in VARIANTS:
            raise PlaygroundError(f"Bilinmeyen varyant: {variant_key}")
        variant = VARIANTS[variant_key]
        text = (text or "").strip()
        if not text:
            raise PlaygroundError("Metin boş.")
        if len(text) > MAX_TEXT_CHARS:
            raise PlaygroundError(f"Metin en fazla {MAX_TEXT_CHARS} karakter olabilir ({len(text)}).")
        if reference is None and profile_slug:
            reference = resolve_profile_reference(profile_slug)
        if variant.needs_reference and reference is None:
            raise PlaygroundError(f"{variant.label} için bir ses profili seçin.")
        if device not in DEVICE_LABELS:
            raise PlaygroundError(f"Bilinmeyen cihaz: {device}")
        if variant.engine == "mock":
            device = "cpu"  # Mock GPU kullanmaz
        transcript = None
        if variant.ultimate:
            transcript = (reference_text or "").strip() or self.get_transcript(profile_slug).strip()
            if not transcript:
                raise PlaygroundError(
                    f"{variant.label} için referans sesin birebir transkriptini yazın "
                    "(referansta ne söyleniyorsa, aynen)."
                )
            if len(transcript) > MAX_TRANSCRIPT_CHARS:
                raise PlaygroundError(f"Transkript en fazla {MAX_TRANSCRIPT_CHARS} karakter olabilir.")
            if profile_slug and reference_text and reference_text.strip():
                self.save_transcript(profile_slug, transcript)  # profil başına hatırlanır
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        spoken, chunks = split_for_engine(text, variant.max_chars, normalize)
        if not chunks:
            raise PlaygroundError("Metinden okunacak bir parça çıkmadı.")

        suffix = "" if device == "cuda" or variant.engine == "mock" else f"_{device}"
        record_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 10**6:06d}_{variant.key}{suffix}"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = self.audio_dir / f"{record_id}.wav"
        started = time.perf_counter()
        reply = self.manager.synthesize(
            variant.engine,
            {
                "chunks": chunks,
                "reference": str(reference) if reference and variant.needs_reference else None,
                "settings": dict(variant.settings),
                "seed": int(seed),
                "out": str(wav_path),
                "reference_text": transcript,
            },
            device=device,
        )
        worker = reply.get("worker", {})
        record = {
            "id": record_id,
            "created_at": utc_now_iso(),
            "variant": variant.key,
            "label": variant.label,
            "engine": variant.engine,
            "model": worker.get("model"),
            "model_source": worker.get("source"),
            "device": worker.get("device"),
            "runtime": {
                **(worker.get("runtime") or {}),
                "dtype": worker.get("dtype"),
                "compile": worker.get("compile"),
                "vram_policy": worker.get("vram_policy"),
            },
            "settings": {
                "overrides": dict(variant.settings),
                "label": variant.settings_label,
                "applied": reply.get("applied_settings", {}),
                "engine_defaults": worker.get("defaults", {}),
                "seed": reply.get("seed", seed),
                "normalize": bool(normalize),
                "max_chars": variant.max_chars,
                "ultimate": variant.ultimate,
            },
            "profile": profile_slug,
            "reference": str(reference) if reference else None,
            "reference_text": transcript,
            "text": text,
            "spoken_text": spoken,
            "chunks": reply.get("chunks", len(chunks)),
            "timing": {
                "generation_seconds": reply.get("seconds"),
                "audio_seconds": reply.get("audio_seconds"),
                "rtf": reply.get("rtf"),
                "model_load_seconds": worker.get("load_seconds") if reply.get("loaded_now") else None,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "chunk_seconds": reply.get("chunk_seconds", []),
                "stage_seconds": reply.get("stage_seconds", {}),
            },
            "vram": {
                "generation_peak_mb": reply.get("vram_peak_mb"),
                "model_load_peak_mb": worker.get("load_vram_peak_mb"),
                **(reply.get("vram") or {}),
            },
            "versions": worker.get("versions", {}),
            "warnings": reply.get("warnings", []),
            "note": worker.get("note"),
            "wav": str(wav_path),
        }
        record["diagnosis"] = diagnose(record)
        write_json_atomic(self.audio_dir / f"{record_id}.json", record)
        return record

    def generate_many(
        self,
        text: str,
        variant_keys: list[str],
        profile_slug: str | None,
        seed: int | None = DEFAULT_SEED,
        normalize: bool = True,
        reference: Path | None = None,
        device: str = "cuda",
        reference_text: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Same text, same seed, several variants; ordered by engine so each model loads once."""
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        engine_order = list(dict.fromkeys(VARIANTS[k].engine for k in VARIANTS))
        current = self.manager.current_engine
        if current in engine_order:  # zaten yüklü motorla başla
            engine_order.remove(current)
            engine_order.insert(0, current)
        keys = sorted(dict.fromkeys(variant_keys), key=lambda k: engine_order.index(VARIANTS[k].engine) if k in VARIANTS else -1)
        records, errors = [], []
        for key in keys:
            try:
                records.append(
                    self.generate(
                        text,
                        key,
                        profile_slug,
                        seed=seed,
                        normalize=normalize,
                        reference=reference,
                        device=device,
                        reference_text=reference_text,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one failing variant does not stop the others
                label = VARIANTS[key].label if key in VARIANTS else key
                errors.append(f"{label}: {exc}")
        return records, errors


def diagnose(record: dict[str, Any]) -> list[str]:
    """Short, evidence-based bottleneck notes for one generation (no guessing)."""
    notes: list[str] = []
    timing = record.get("timing", {}) or {}
    vram = record.get("vram", {}) or {}
    device = record.get("device")
    rtf = timing.get("rtf")
    seconds = timing.get("generation_seconds") or 0
    if record.get("engine") != "mock" and device == "cpu":
        notes.append("CPU'da üretildi (bilerek seçildi): GPU'ya göre çok daha yavaş beklenir.")
    if device == "cuda" and isinstance(rtf, (int, float)) and rtf > 1:
        notes.append(f"RTF {rtf} > 1: GPU'da bile gerçek zamandan yavaş; uzun parçalar veya yeniden denemeler olabilir.")
    total = vram.get("device_total_mb")
    reserved = vram.get("reserved_mb")
    used = vram.get("device_used_mb")
    if isinstance(total, (int, float)) and total > 0:
        if isinstance(used, (int, float)) and used / total >= 0.97:
            notes.append(
                f"VRAM tamamen dolu (%{used / total * 100:.0f}): Windows belleği sistem RAM'ine taşıyabilir ve üretim "
                "10 kat ve daha fazla yavaşlar. GPU kullanan diğer uygulamaları (canlı duvar kağıdı, tarayıcı, oyun istemcisi) kapatın."
            )
        elif isinstance(used, (int, float)) and used / total >= 0.9:
            notes.append(f"VRAM sınırına yakın: GPU belleğinin %{used / total * 100:.0f}'i dolu.")
        if isinstance(used, (int, float)) and isinstance(reserved, (int, float)) and used - reserved > 1500:
            notes.append(
                f"Bu süreç dışında ~{used - reserved:,.0f} MB GPU belleği kullanımda (başka uygulama veya CUDA bağlamı).".replace(",", ".")
            )
    prompt = (timing.get("stage_seconds") or {}).get("prompt_seconds")
    if isinstance(prompt, (int, float)) and seconds and prompt / seconds >= 0.25:
        notes.append(f"Referans/prompt kodlaması sürenin %{prompt / seconds * 100:.0f}'i ({prompt} sn).")
    if (timing.get("stage_seconds") or {}).get("prompt_cache_hits"):
        notes.append("Referans kodlaması önbellekten geldi (aynı referans; kodlama atlandı).")
    if timing.get("model_load_seconds") is not None:
        notes.append(f"Bu üretimden önce model yüklendi ({timing['model_load_seconds']} sn; RTF'ye dahil değil).")
    return notes


def describe(record: dict[str, Any]) -> str:
    """Markdown summary of one generation (model, settings, time, VRAM)."""
    timing = record.get("timing", {})
    vram = record.get("vram", {})
    settings = record.get("settings", {})

    def mb(value: Any) -> str:
        return f"{value:,.0f} MB".replace(",", ".") if isinstance(value, (int, float)) else "—"

    lines = [
        f"**{record.get('label')}** · `{record.get('id')}`",
        f"- Model: `{record.get('model')}` · cihaz: **{DEVICE_LABELS.get(record.get('device'), record.get('device'))}**",
        f"- Ayar: {settings.get('label')} · uygulanan: `{json.dumps(settings.get('applied') or settings.get('overrides') or {}, ensure_ascii=False)}` · tohum {settings.get('seed')}",
        f"- Profil: {record.get('profile') or '—'} · parça: {record.get('chunks')} · normalizasyon: {'açık' if settings.get('normalize') else 'kapalı'}",
        f"- Süre: üretim {timing.get('generation_seconds')} sn · ses {timing.get('audio_seconds')} sn · RTF {timing.get('rtf')}"
        + (f" · model yükleme {timing.get('model_load_seconds')} sn" if timing.get("model_load_seconds") is not None else ""),
    ]
    runtime = record.get("runtime") or {}
    if runtime:
        parts = [f"torch {runtime.get('torch')}"]
        if runtime.get("gpu"):
            parts.append(f"{runtime['gpu']} (sm {runtime.get('compute_capability')})")
        elif record.get("engine") != "mock":
            parts.append(f"CPU · {runtime.get('cpu_threads')} iş parçacığı")
        if runtime.get("dtype"):
            parts.append(f"dtype {runtime['dtype']}")
        if runtime.get("compile"):
            parts.append(f"torch.compile {runtime['compile']}")
        lines.append("- Çalışma ortamı: " + " · ".join(parts))
    if settings.get("ultimate"):
        lines.append("- Ultimate Cloning: referans + transkript (prompt devamı)")
    if vram.get("cuda"):
        lines.append(
            f"- VRAM: üretim tepe {mb(vram.get('generation_peak_mb'))} · ayrılmış {mb(vram.get('reserved_mb'))}"
            f" · GPU kullanımı {mb(vram.get('device_used_mb'))}/{mb(vram.get('device_total_mb'))}"
        )
    else:
        lines.append("- VRAM: CUDA yok (CPU/Mock)")
    chunk_times = timing.get("chunk_seconds") or []
    if len(chunk_times) > 1:
        lines.append("- Parça süreleri (sn): " + ", ".join(str(t) for t in chunk_times))
    for note in record.get("diagnosis") or diagnose(record):
        lines.append(f"- Darboğaz: {note}")
    for warning in record.get("warnings") or []:
        lines.append(f"- Uyarı: {warning}")
    if record.get("note"):
        lines.append(f"- Not: {record['note']}")
    return "\n".join(lines)
