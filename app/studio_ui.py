# -*- coding: utf-8 -*-
"""VoxForge Studio V1 tab (VF-006).

Thin Gradio layer over ``voxforge.studio.service.StudioService``. The existing
XTTS demo is untouched; both tabs share the same cached engine (and its
inference lock).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gradio as gr

from voxforge.studio.service import STATE_LABELS, StudioError, StudioService

TABLE_HEADERS = ["#", "Durum", "Orijinal metin", "Okunacak metin", "Süre (sn)", "Hata"]
REFRESH_SECONDS = 2.0
NO_SELECTION = "Bir segment seçmek için tablodaki satıra tıklayın."


def _short(text: str | None, limit: int = 90) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_transformations(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_Dönüşüm yok: okunacak metin orijinalle aynı._"
    lines = ["| Kural | Orijinal | Okunuş |", "|---|---|---|"]
    rule_names = {
        "number": "sayı", "percent": "yüzde", "time": "saat", "date": "tarih",
        "money": "para", "lexicon": "sözlük", "manual": "elle düzenlendi",
    }
    for item in items:
        original = (item.get("original") or "—").replace("|", "/")
        spoken = (item.get("spoken") or "").replace("|", "/")
        lines.append(f"| {rule_names.get(item.get('rule'), item.get('rule'))} | `{original}` | {spoken} |")
    return "\n".join(lines)


def build_studio(
    service: StudioService,
    *,
    engine_label: str,
    profile_choices: Callable[[], list[tuple[str, str]]],
    resolve_reference: Callable[[str | None, str | None], Path | None],
    preset_choices: list[str],
    preset_kwargs: dict[str, dict[str, Any]],
    default_preset: str,
) -> gr.Blocks:
    def project_choices() -> list[tuple[str, str]]:
        return service.list_projects()

    def progress_text(slug: str | None, info: dict[str, Any] | None) -> str:
        if not slug or info is None:
            return "Proje seçilmedi. Mevcut bir projeyi açın veya yeni proje oluşturun."
        project = service.open(slug)
        settings = project.settings
        counts = info["counts"]
        done = counts.get("done", 0)
        parts = [f"**{project.name}** · motor `{settings.engine_id}` · {done}/{info['total']} segment hazır"]
        details = [
            f"{STATE_LABELS[state]}: {counts[state]}"
            for state in ("running", "queued", "failed", "cancelled", "pending")
            if counts.get(state)
        ]
        if details:
            parts.append(" · ".join(details))
        parts.append(f"Toplam süre: {info['duration_s']:.1f} sn")
        parts.append("⏳ Üretim çalışıyor…" if info["running"] else "Üretim durdu.")
        if info["worker_error"]:
            parts.append(f"⚠️ Çalıştırıcı hatası: {info['worker_error']}")
        if counts.get("failed"):
            parts.append("Hatalı segmentler tabloda; düzeltip ya da doğrudan **Eksikleri üret** ile yeniden deneyebilirsiniz.")
        return "\n\n".join(parts)

    def table(info: dict[str, Any] | None) -> tuple[list[list[Any]], list[int]]:
        if info is None:
            return [], []
        rows, ids = [], []
        for item in info["segments"]:
            ids.append(item["id"])
            rows.append(
                [
                    item["ord"] + 1,
                    STATE_LABELS[item["state"]],
                    _short(item["source_text"]),
                    _short(item["spoken_text"]),
                    round(item["duration_s"], 2) if item["duration_s"] else "",
                    _short(item["error"], 60) if item["error"] else "",
                ]
            )
        return rows, ids

    def view(slug: str | None):
        # Tek tutarlı anlık görüntü: çalışma durumu segmentlerden önce okunur. Böylece
        # running=False yazıldığında tablo son segmenti de içerir ve son yenileme kaçmaz.
        info = service.snapshot(slug) if slug else None
        rows, ids = table(info)
        return progress_text(slug, info), rows, ids, bool(info and info["running"])

    def segment_view(slug: str | None, segment_id: int | None):
        if not slug or segment_id is None:
            return NO_SELECTION, "", "", "", None
        item = service.segment(slug, segment_id)
        header = f"**Segment {item['ord'] + 1}** · {STATE_LABELS[item['state']]}"
        if item["error"]:
            header += f"\n\n⚠️ {item['error']}"
        return (
            header,
            item["source_text"],
            item["spoken_text"],
            _format_transformations(item["transformations"]),
            item["audio_path"],
        )

    with gr.Blocks(title="VoxForge Studio") as studio:
        gr.Markdown("# VoxForge Studio")
        gr.Markdown(
            f"Metni projeye dönüştürün, segmentleri inceleyin ve yalnız eksik ya da değişen parçaları üretin. "
            f"Etkin motor: `{engine_label}`. Projeler yerel `studio_projects/` klasöründe tutulur."
        )
        slug_state = gr.State(None)
        ids_state = gr.State([])
        selected_state = gr.State(None)
        running_state = gr.State(False)

        with gr.Row():
            project_dropdown = gr.Dropdown(label="Proje", choices=project_choices(), value=None, scale=4)
            refresh_projects = gr.Button("Listeyi yenile", scale=1)

        with gr.Accordion("Yeni proje", open=False):
            new_name = gr.Textbox(label="Proje adı")
            new_text = gr.Textbox(label="Metin", lines=10, placeholder="Seslendirilecek Türkçe metni yapıştırın.")
            with gr.Row():
                new_profile = gr.Dropdown(label="Ses profili", choices=profile_choices(), value="")
                new_upload = gr.Audio(label="veya referans ses yükle", type="filepath")
            new_preset = gr.Dropdown(label="Üretim modu", choices=preset_choices, value=default_preset)
            new_lexicon = gr.Textbox(
                label="Telaffuz sözlüğü (isteğe bağlı)",
                lines=3,
                placeholder="Her satıra bir giriş: VoxForge = voks forc",
            )
            gr.Markdown(
                "Referans önceliği demo ile aynıdır: profil, sonra yüklenen ses, sonra varsayılan referans. "
                "Seçilen referans proje klasörüne kopyalanır; profil sonradan değişse de proje etkilenmez."
            )
            create_button = gr.Button("Projeyi oluştur", variant="primary")
            create_status = gr.Markdown()

        progress_md = gr.Markdown("Proje seçilmedi.")
        with gr.Row():
            start_button = gr.Button("Eksikleri üret", variant="primary")
            stop_button = gr.Button("Durdur")
            preview_button = gr.Button("Tümünü birleştir (önizleme)")
        action_status = gr.Markdown()
        preview_audio = gr.Audio(label="Önizleme", type="filepath", interactive=False, visible=False)

        segment_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["number", "str", "str", "str", "number", "str"],
            interactive=False,
            wrap=True,
            label="Segmentler",
        )

        with gr.Group():
            segment_header = gr.Markdown(NO_SELECTION)
            source_box = gr.Textbox(label="Orijinal metin", lines=3)
            save_source = gr.Button("Orijinali kaydet (okunuş yeniden hesaplanır)")
            spoken_box = gr.Textbox(label="Okunacak metin", lines=3)
            save_spoken = gr.Button("Okunuşu elle kaydet")
            transformations_md = gr.Markdown()
            segment_audio = gr.Audio(label="Segment sesi", type="filepath", interactive=False)
            segment_status = gr.Markdown()

        timer = gr.Timer(REFRESH_SECONDS)

        view_outputs = [progress_md, segment_table, ids_state, running_state]
        segment_outputs = [segment_header, source_box, spoken_box, transformations_md, segment_audio]

        def on_project_change(slug):
            if not slug:
                return (None, *view(None), None, *segment_view(None, None))
            try:
                return (slug, *view(slug), None, *segment_view(slug, None))
            except StudioError as exc:
                return (None, f"⚠️ {exc}", [], [], False, None, *segment_view(None, None))

        project_dropdown.change(
            on_project_change,
            inputs=project_dropdown,
            outputs=[slug_state, *view_outputs, selected_state, *segment_outputs],
        )
        refresh_projects.click(lambda: gr.update(choices=project_choices()), outputs=project_dropdown)

        def on_create(name, text, profile_slug, upload, preset_label, lexicon_text):
            try:
                reference = resolve_reference(profile_slug or None, upload)
                slug = service.create(
                    name, text, reference, preset_kwargs.get(preset_label, {}), lexicon_text
                )
            except StudioError as exc:
                return f"⚠️ {exc}", gr.update()
            except Exception as exc:  # noqa: BLE001
                return f"⚠️ Proje oluşturulamadı: {exc}", gr.update()
            count = len(service.segments(slug))
            return (
                f"Proje oluşturuldu: **{name}** ({count} segment).",
                gr.update(choices=project_choices(), value=slug),
            )

        create_button.click(
            on_create,
            inputs=[new_name, new_text, new_profile, new_upload, new_preset, new_lexicon],
            outputs=[create_status, project_dropdown],
        )

        def on_start(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None)
            try:
                queued = service.start(slug)
            except StudioError as exc:
                return f"⚠️ {exc}", *view(slug)
            except Exception as exc:  # noqa: BLE001 - e.g. model load failure
                return f"⚠️ Motor başlatılamadı: {exc}", *view(slug)
            message = (
                f"Üretim başladı: {queued} segment kuyrukta."
                if queued
                else "Tüm segmentler güncel; üretilecek bir şey yok."
            )
            return message, *view(slug)

        start_button.click(on_start, inputs=slug_state, outputs=[action_status, *view_outputs])

        def on_stop(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None)
            cancelled = service.stop(slug)
            return f"Durduruluyor: mevcut segment bitince duracak. {cancelled} segment kuyruktan çıkarıldı.", *view(slug)

        stop_button.click(on_stop, inputs=slug_state, outputs=[action_status, *view_outputs])

        def on_preview(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin.", gr.update(visible=False)
            try:
                path = service.build_preview(slug)
            except StudioError as exc:
                return f"⚠️ {exc}", gr.update(visible=False)
            return (
                f"Önizleme hazır ({service.preview_seconds(path):.1f} sn): `{path.name}`",
                gr.update(value=str(path), visible=True),
            )

        preview_button.click(on_preview, inputs=slug_state, outputs=[action_status, preview_audio])

        def on_select(slug, ids, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not slug or row is None or row >= len(ids):
                return None, *segment_view(None, None)
            segment_id = ids[row]
            return segment_id, *segment_view(slug, segment_id)

        segment_table.select(on_select, inputs=[slug_state, ids_state], outputs=[selected_state, *segment_outputs])

        def on_save(kind):
            def handler(slug, segment_id, text):
                if not slug or segment_id is None:
                    return "⚠️ Önce bir segment seçin.", *view(slug), *segment_view(slug, segment_id)
                try:
                    if kind == "source":
                        service.update_source(slug, segment_id, text)
                        message = "Orijinal metin kaydedildi; okunuş yeniden hesaplandı."
                    else:
                        service.set_spoken(slug, segment_id, text)
                        message = "Okunacak metin elle kaydedildi."
                except StudioError as exc:
                    return f"⚠️ {exc}", *view(slug), *segment_view(slug, segment_id)
                message += " Yalnız bu segment yeniden üretilecek; **Eksikleri üret** ile başlatın."
                return message, *view(slug), *segment_view(slug, segment_id)

            return handler

        save_source.click(
            on_save("source"),
            inputs=[slug_state, selected_state, source_box],
            outputs=[segment_status, *view_outputs, *segment_outputs],
        )
        save_spoken.click(
            on_save("spoken"),
            inputs=[slug_state, selected_state, spoken_box],
            outputs=[segment_status, *view_outputs, *segment_outputs],
        )

        def on_tick(slug, was_running, segment_id):
            # Yalnız üretim sürerken (ve bittiği ilk turda) yenile; boşta veritabanını yormaz.
            # Düzenleme kutularına dokunulmaz ki kullanıcının yazdığı metin ezilmesin.
            if not slug or (not was_running and not service.is_running(slug)):
                return (gr.skip(),) * 6
            header, _source, _spoken, _transforms, audio = segment_view(slug, segment_id)
            return *view(slug), header, audio

        timer.tick(
            on_tick,
            inputs=[slug_state, running_state, selected_state],
            outputs=[*view_outputs, segment_header, segment_audio],
        )

    return studio
