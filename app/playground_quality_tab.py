# -*- coding: utf-8 -*-
"""Voice Playground "Kalite ve take'ler" tab (VF-018).

Gradio wiring only; the logic lives in ``voxforge.playground.quality`` (Playground
adapter) and ``voxforge.quality`` (Quality Director + Voice Takes, UI independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import gradio as gr

from voxforge.playground.quality import PlaygroundQuality
from voxforge.quality.checks import describe_report, summarize
from voxforge.quality.takes import DEFAULT_ALTERNATIVES, MAX_ALTERNATIVES, TakeError, rank_key, selection_stack

TAKE_SLOTS = 4
TAKE_HEADERS = ["Take", "Köken", "Tohum", "Karar", "CER", "Benzerlik", "Bulgular", "Durum"]
WHOLE_TEXT = "__all__"
INTRO = (
    "Üretimi **Quality Director** ile denetle: Whisper (large-v3, CPU; VRAM kullanmaz) metni kelime kelime karşılaştırır "
    "(eksik / tekrarlanan / metin dışı / farklı kelime, kesilme), ses ölçümleri süreyi, hızı, sessizlikleri, seviyeyi ve "
    "kırpılmayı denetler. **ASR farkı kesin TTS hatası değildir**: her bulgu güven düzeyi ve kanıt kaynağıyla gösterilir; "
    "*şüpheli* sonuçta hiçbir şey kendiliğinden değişmez. Alternatifler aynı ayarlarla **farklı tohumlarla** üretilir; her "
    "take ayarı, sesi ve kalite raporuyla saklanır ve silinmez. En iyi take *önerilir*, seçim yalnız sende; önceki seçime "
    "dönülebilir. Çok cümleli metinde yalnız sorunlu cümle (⚠️) için alternatif üretilebilir."
)


def take_rows(group: dict[str, Any] | None) -> list[list[Any]]:
    if not group:
        return []
    rows = []
    for take in group["takes"]:
        report = take.get("quality") or {}
        metrics = report.get("metrics") or {}
        findings = [f["label"] for f in report.get("findings", []) if f["confidence"] != "low" or f["severity"] == "major"]
        state = []
        if take["take_id"] == group.get("selected"):
            state.append("SEÇİLİ")
        if take["take_id"] == group.get("recommended"):
            state.append("önerilen")
        rows.append([
            take["take_id"], "orijinal" if take.get("origin") == "original" else "alternatif", take.get("seed"),
            report.get("verdict_label", "denetlenmedi"), metrics.get("cer"), take.get("similarity"),
            ", ".join(dict.fromkeys(findings)) or "—", " · ".join(state),
        ])
    return rows


def slot_takes(group: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Selected take first, then the best ranked others (side-by-side listening)."""
    if not group:
        return []
    takes = group["takes"]
    selected = [t for t in takes if t["take_id"] == group.get("selected")]
    others = sorted((t for t in takes if t["take_id"] != group.get("selected")), key=lambda t: rank_key(t) if t.get("quality") else (9,))
    return (selected + others)[:TAKE_SLOTS]


def describe_take(take: dict[str, Any], group: dict[str, Any]) -> str:
    timing = take.get("timing") or {}
    head = f"**{take['take_id']}** · tohum {take.get('seed')} · {summarize(take.get('quality'))}"
    if take["take_id"] == group.get("selected"):
        head = "✅ " + head + " · **seçili**"
    elif take["take_id"] == group.get("recommended"):
        head += " · _önerilen_"
    lines = [head, f"- Ayar: {(take.get('settings') or {}).get('label')} · ses {timing.get('audio_seconds')} sn · RTF {timing.get('rtf')}"]
    if take.get("similarity") is not None:
        lines.append(f"- Referans benzerliği (ECAPA): {take['similarity']}")
    for item in (take.get("quality") or {}).get("findings", []):
        if item["confidence"] != "low" or item["severity"] == "major":
            lines.append(f"- {item['label']}: {item['message']}")
    return "\n".join(lines)


def sentence_choices(report: dict[str, Any] | None) -> list[tuple[str, str]]:
    choices = [("Tüm metin", WHOLE_TEXT)]
    sentences = (report or {}).get("sentences") or []
    problems = set((report or {}).get("problem_sentences") or [])
    if len(sentences) > 1:
        for index, sentence in enumerate(sentences):
            short = sentence if len(sentence) <= 50 else sentence[:47] + "..."
            choices.append((f"{'⚠️ ' if index in problems else ''}Cümle {index + 1}: {short}", str(index)))
    return choices


@dataclass
class QualityTab:
    quality: PlaygroundQuality
    history_choices: Callable[[], list[tuple[str, str]]]
    record: Any = None
    report: Any = None
    sentence: Any = None
    group: Any = None
    table: Any = None
    take_components: list[Any] = field(default_factory=list)
    take: Any = None
    message: Any = None

    # -- layout -------------------------------------------------------------------------
    def build(self) -> None:
        gr.Markdown(INTRO)
        with gr.Row():
            self.record = gr.Dropdown(self.history_choices(), label="Üretim", value=None, scale=4)
            refresh = gr.Button("Listeyi yenile", size="sm", scale=1)
        with gr.Row():
            force = gr.Checkbox(value=False, label="Önbelleği yok say, yeniden denetle")
            check = gr.Button("Kalite denetimi yap", variant="secondary")
        self.report = gr.Markdown()
        with gr.Row():
            self.sentence = gr.Dropdown([("Tüm metin", WHOLE_TEXT)], value=WHOLE_TEXT, label="Alternatif üretilecek bölüm (⚠️ = sorunlu cümle)", scale=3)
            count = gr.Slider(1, MAX_ALTERNATIVES, value=DEFAULT_ALTERNATIVES, step=1, label="Alternatif sayısı (farklı tohum)", scale=2)
            generate = gr.Button("Alternatif üret", variant="primary", scale=1)
        self.group = gr.Dropdown([], label="Take grubu", value=None)
        self.table = gr.Dataframe(headers=TAKE_HEADERS, interactive=False, wrap=True, label="Take'ler (hiçbiri silinmez)")
        with gr.Row(equal_height=False):
            for _ in range(TAKE_SLOTS):
                with gr.Column(min_width=220):
                    self.take_components += [gr.Audio(type="filepath", interactive=False, visible=False), gr.Markdown(visible=False)]
        with gr.Row():
            self.take = gr.Dropdown([], label="Take", value=None)
            select = gr.Button("Bu take'i seç", variant="primary")
            revert = gr.Button("Önceki seçime dön")
        self.message = gr.Markdown()

        refresh.click(self.record_view, self.record, self.outputs)
        self.record.input(self.record_view, self.record, self.outputs)
        check.click(self.on_check, [self.record, force], self.outputs)
        generate.click(self.on_generate, [self.record, self.sentence, count], self.outputs)
        self.group.input(self.group_view, self.group, self.group_outputs)
        select.click(self.on_select, [self.group, self.take], self.group_outputs)
        revert.click(self.on_revert, self.group, self.group_outputs)

    @property
    def outputs(self) -> list[Any]:
        return [self.record, self.report, self.sentence, self.group, *self.group_outputs]

    @property
    def group_outputs(self) -> list[Any]:
        return [self.table, *self.take_components, self.take, self.message]

    # -- views --------------------------------------------------------------------------
    def groups_of(self, record_id: str | None) -> list[tuple[str, str]]:
        if not record_id:
            return []
        prefix = self.quality.group_id(record_id)
        choices = []
        for group in self.quality.store.list():
            if group["group_id"] == prefix or group["group_id"].startswith(prefix + "-s"):
                sentence = (group.get("source") or {}).get("sentence")
                label = "Tüm metin" if sentence is None else f"Cümle {sentence + 1}: {group['text'][:40]}"
                choices.append((f"{label} · {len(group['takes'])} take", group["group_id"]))
        return sorted(choices, key=lambda item: item[1])

    def group_view(self, group_id: str | None, note: str = "") -> list[Any]:
        group = None
        if group_id:
            try:
                group = self.quality.store.load(group_id)
            except TakeError as exc:
                note = str(exc)
        updates: list[Any] = [take_rows(group)]
        shown = slot_takes(group)
        for index in range(TAKE_SLOTS):
            if index < len(shown):
                take = shown[index]
                updates += [
                    gr.update(value=take["wav"], label=f"{take['take_id']} · tohum {take.get('seed')}", visible=True),
                    gr.update(value=describe_take(take, group), visible=True),
                ]
            else:
                updates += [gr.update(value=None, visible=False), gr.update(value="", visible=False)]
        take_ids = [t["take_id"] for t in (group or {}).get("takes", [])]
        updates.append(gr.update(choices=take_ids, value=(group or {}).get("recommended") or (take_ids[0] if take_ids else None)))
        if not note and group and len(selection_stack(group["selection_history"])) > 1:
            note = f"Seçili: **{group['selected']}** · önceki seçime dönülebilir."
        updates.append(note)
        return updates

    def record_view(self, record_id: str | None, group_id: str | None = None, note: str = "", report: dict[str, Any] | None = None) -> list[Any]:
        if report is None and record_id:
            report = self.quality.report(record_id)
        groups = self.groups_of(record_id)
        if group_id is None and groups:
            group_id = groups[0][1]
        return [
            gr.update(choices=self.history_choices(), value=record_id),
            describe_report(report) if record_id else "",
            gr.update(choices=sentence_choices(report), value=WHOLE_TEXT),
            gr.update(choices=groups, value=group_id),
            *self.group_view(group_id, note),
        ]

    # -- events -------------------------------------------------------------------------
    def on_check(self, record_id: str | None, force: bool) -> list[Any]:
        if not record_id:
            return self.record_view(None, note="Önce bir üretim seçin.")
        try:
            report = self.quality.check(record_id, force=bool(force))
        except Exception as exc:  # noqa: BLE001
            return self.record_view(record_id, note=f"Denetim başarısız: {exc}")
        return self.record_view(record_id, report=report)

    def on_generate(self, record_id: str | None, sentence_value: str | None, count: Any) -> list[Any]:
        if not record_id:
            return self.record_view(None, note="Önce bir üretim seçin.")
        sentence = None if sentence_value in (None, WHOLE_TEXT) else int(sentence_value)
        try:
            group, errors = self.quality.alternatives(record_id, int(count), sentence)
        except Exception as exc:  # noqa: BLE001
            return self.record_view(record_id, note=f"Alternatif üretilemedi: {exc}")
        note = f"{int(count) - len(errors)} alternatif eklendi; öneri: **{group.get('recommended')}** (seçim sende)."
        if errors:
            note += "  \n" + "  \n".join(f"Hata — {e}" for e in errors)
        return self.record_view(record_id, group["group_id"], note)

    def on_select(self, group_id: str | None, take_id: str | None) -> list[Any]:
        if not group_id or not take_id:
            return self.group_view(group_id, "Take seçin.")
        try:
            group = self.quality.select(group_id, take_id)
        except TakeError as exc:
            return self.group_view(group_id, str(exc))
        return self.group_view(group_id, f"Seçildi: **{group['selected']}** (önceki seçim geçmişte saklı).")

    def on_revert(self, group_id: str | None) -> list[Any]:
        if not group_id:
            return self.group_view(None, "Take grubu yok.")
        try:
            group = self.quality.revert(group_id)
        except TakeError as exc:
            return self.group_view(group_id, str(exc))
        return self.group_view(group_id, f"Önceki seçime dönüldü: **{group['selected']}**.")

    def after_generation(self, ids: list[str] | None, enabled: bool, retake: bool, count: Any) -> list[Any]:
        """Runs after a Playground generation (``.then``): the audio is already shown."""
        record_id = (list(ids or []) or [None])[0]
        record = self.quality.service.get(record_id) if record_id else None
        if not enabled or record is None or record.get("engine") == "mock":
            return ["", *self.record_view(record_id)]
        try:
            result = self.quality.after_generation(record, True, bool(retake), int(count))
        except Exception as exc:  # noqa: BLE001 - denetim üretimi geri almaz
            return [f"Kalite denetimi yapılamadı: {exc}", *self.record_view(record_id)]
        report = result["report"]
        note = describe_report(report)
        group_id = None
        if result["auto_retake"]:
            group_id = result["group"]["group_id"]
            note += (
                f"\n\n**Yüksek güvenli sorun:** {int(count) - len(result['errors'])} alternatif üretildi; öneri "
                f"**{result['group'].get('recommended')}**. Seçim için *Kalite ve take'ler* sekmesi."
            )
        elif report and report.get("verdict") == "suspect":
            note += "\n\nŞüpheli: dinleyin; isterseniz *Kalite ve take'ler* sekmesinden alternatif üretin."
        return [note, *self.record_view(record_id, group_id, report=report)]
