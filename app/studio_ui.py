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

from voxforge.quality.checks import describe_report, summarize
from voxforge.studio.service import NARRATOR_LABEL, STATE_LABELS, StudioError, StudioService

TABLE_HEADERS = ["#", "Bölüm", "Durum", "Ses", "Orijinal metin", "Okunacak metin", "Süre (sn)", "Hata"]
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
    vox_presets: dict[str, tuple[str, dict[str, Any]]] | None = None,
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
                    item["chapter_title"],
                    STATE_LABELS[item["state"]],
                    item["voice_name"] or NARRATOR_LABEL,
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
        header = (
            f"**Segment {item['ord'] + 1}** · {STATE_LABELS[item['state']]}"
            f" · ses: {item['voice_name'] or NARRATOR_LABEL}"
        )
        if item["error"]:
            header += f"\n\n⚠️ {item['error']}"
        return (
            header,
            item["source_text"],
            item["spoken_text"],
            _format_transformations(item["transformations"]),
            item["audio_path"],
        )

    def take_panel(slug, segment_id, token=None):
        if not slug or segment_id is None:
            return gr.update(choices=[], value=None), "", None, gr.update(interactive=False), gr.update(interactive=False)
        item = service.take_view(slug, segment_id)
        current = item["selected_take"]
        prefix = item["cache_key"] + ":"
        choices = [
            (f"{'İlk take' if i == 0 else f'Take {i + 1}'}"
             f"{' · seçili' if current == take['id'] else ''} · {summarize(take['quality'])}", prefix + str(take["id"]))
            for i, take in enumerate(item["takes"]) if take["path"]
        ]
        value = token if token in {value for _, value in choices} else prefix + str(current)
        candidate = next((take for take in item["takes"] if value == prefix + str(take["id"])), None)
        job = item["job"]
        message = STATE_LABELS.get(job["status"], "") if job and job["status"] != "done" else ""
        if job and job["error"]:
            message += f" · {job['error']}"
        report = describe_report(candidate["quality"]) if candidate else ""
        return (
            gr.update(choices=choices, value=value if candidate and candidate["path"] else None),
            "\n\n".join(part for part in (message, report) if part),
            candidate["path"] if candidate and candidate["id"] != current else None,
            gr.update(interactive=bool(candidate and candidate["path"] and candidate["id"] != current)),
            gr.update(interactive=item["can_undo"]),
        )

    def preset_for(engine_id, label):
        if engine_id == "voxcpm2":
            return (vox_presets or {}).get(label, ("", {}))[1]
        return preset_kwargs.get(label, {})

    def preset_update(engine_id):
        if engine_id == "voxcpm2" and vox_presets:
            choices = [(name, key) for key, (name, _) in vox_presets.items()]
            return gr.update(choices=choices, value="normal", visible=True)
        return gr.update(choices=preset_choices, value=default_preset, visible=engine_id != "mock")

    with gr.Blocks(title="VoxForge Studio") as studio:
        gr.Markdown("# VoxForge Studio")
        gr.Markdown(
            "Metni projeye dönüştürün, segmentleri inceleyin ve yalnız eksik ya da değişen parçaları üretin. "
            "Kaynak belgeler, projeler ve WAV/MP3 çıktıları yerelde tutulur."
        )
        if not vox_presets:
            gr.Markdown("VoxCPM2 üretimi için Voice Playground uygulamasında **Studio** sekmesini açın.")
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
            new_document = gr.File(
                label="veya kitap dosyası (TXT, DOCX, EPUB)", type="filepath", file_types=[".txt", ".docx", ".epub"],
            )
            gr.Markdown("TXT’de `Bölüm 1: Başlık` veya `# Başlık` kullanın. Paragrafları boş satırla ayırın.")
            new_engine = gr.Dropdown(
                label="Studio motoru", choices=[("VoxCPM2 (CUDA)", "voxcpm2"), ("XTTS-v2 (CUDA)", "xtts"), ("Mock", "mock")],
                value="voxcpm2" if vox_presets else None, visible=bool(vox_presets),
            )
            with gr.Row():
                new_profile = gr.Dropdown(label="Ses profili", choices=profile_choices(), value="")
                new_upload = gr.Audio(label="veya referans ses yükle", type="filepath")
            new_preset = gr.Dropdown(
                label="Üretim modu",
                choices=[(name, key) for key, (name, _) in vox_presets.items()] if vox_presets else preset_choices,
                value="normal" if vox_presets else default_preset,
            )
            new_lexicon = gr.Textbox(
                label="Telaffuz sözlüğü (isteğe bağlı)",
                lines=3,
                placeholder="! New York = Niyu York   (~ ile harf büyüklüğünden bağımsız eşleştir)",
            )
            gr.Markdown(
                "Referans önceliği demo ile aynıdır: profil, sonra yüklenen ses, sonra varsayılan referans. "
                "Seçilen referans proje klasörüne kopyalanır; profil sonradan değişse de proje etkilenmez."
            )
            create_button = gr.Button("Projeyi oluştur", variant="primary")
            create_status = gr.Markdown()

        progress_md = gr.Markdown("Proje seçilmedi.")
        with gr.Row():
            chapter_scope = gr.Dropdown(label="Üretim ve dışa aktarma kapsamı", choices=[], scale=2)
            start_button = gr.Button("Eksikleri ve hatalıları üret / devam et", variant="primary")
            stop_button = gr.Button("Durdur")
        with gr.Row():
            export_format = gr.Dropdown(label="Ses biçimi", choices=["WAV", "MP3"], value="WAV")
            preview_button = gr.Button("Dışa aktar")
        action_status = gr.Markdown()
        preview_audio = gr.Audio(label="Ses çıktısı", type="filepath", interactive=False, visible=False)

        with gr.Accordion("Proje telaffuz sözlüğü", open=False):
            lexicon_box = gr.Textbox(label="Girişler", lines=5, placeholder="! Özel İsim = okunuş")
            save_lexicon = gr.Button("Sözlüğü kaydet")
            lexicon_status = gr.Markdown()

        segment_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["number", "str", "str", "str", "str", "str", "number", "str"],
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

        with gr.Accordion("Segment kalitesi ve take'ler", open=True):
            take_choice = gr.Dropdown(label="Take", choices=[])
            take_quality = gr.Markdown()
            take_audio = gr.Audio(label="Alternatif take sesi", type="filepath", interactive=False)
            with gr.Row():
                take_generate = gr.Button("Bir alternatif üret", visible=bool(vox_presets))
                take_check = gr.Button("Kaliteyi denetle")
                take_select = gr.Button("Take'i kullan", interactive=False)
                take_undo = gr.Button("Take seçimini geri al", interactive=False)
            take_status = gr.Markdown()

        with gr.Accordion("Karakterler ve sesler", open=False):
            gr.Markdown(
                "Atanmamış segmentler **Anlatıcı** (projenin referans sesi) ile üretilir. "
                "Karakter ekleyip segmentlere atayın; ses değişen segmentler yeniden üretilir."
            )
            voices_md = gr.Markdown()
            with gr.Row():
                voice_name = gr.Textbox(label="Karakter adı", scale=2)
                voice_profile = gr.Dropdown(label="Ses profili", choices=profile_choices(), value="", scale=2)
                voice_upload = gr.Audio(label="veya referans ses", type="filepath", scale=2)
                voice_preset = gr.Dropdown(label="Üretim modu", choices=preset_choices, value=default_preset, scale=1)
            add_voice_button = gr.Button("Karakter ekle")
            with gr.Row():
                assign_numbers = gr.Textbox(label="Segmentler", placeholder="3, 5-8", scale=2)
                assign_voice = gr.Dropdown(label="Ses", choices=[(NARRATOR_LABEL, "")], value="", scale=2)
                assign_button = gr.Button("Ata", scale=1)
            with gr.Row():
                suggest_button = gr.Button("Diyalog önerilerini göster")
                strip_labels = gr.Checkbox(label="'Ad:' etiketini okunuştan çıkar", value=True)
                apply_button = gr.Button("Eşleşen önerileri uygula")
            suggestions_md = gr.Markdown()
            voice_status = gr.Markdown()

        timer = gr.Timer(REFRESH_SECONDS)

        view_outputs = [progress_md, segment_table, ids_state, running_state]
        segment_outputs = [segment_header, source_box, spoken_box, transformations_md, segment_audio]
        take_outputs = [take_choice, take_quality, take_audio, take_select, take_undo]

        def on_project_change(slug):
            if not slug:
                return (None, *view(None), None, NO_SELECTION, gr.update(value="", interactive=False), "", "", None)
            try:
                editable = service.open(slug).source_format is None
                return (slug, *view(slug), None, NO_SELECTION, gr.update(value="", interactive=editable), "", "", None)
            except StudioError as exc:
                return (None, f"⚠️ {exc}", [], [], False, None, *segment_view(None, None))

        project_dropdown.change(
            on_project_change,
            inputs=project_dropdown,
            outputs=[slug_state, *view_outputs, selected_state, *segment_outputs],
        )
        project_dropdown.change(
            lambda: (*take_panel(None, None), gr.update(value=None, visible=False)),
            outputs=[*take_outputs, preview_audio],
        )
        refresh_projects.click(lambda: gr.update(choices=project_choices()), outputs=project_dropdown)
        new_engine.change(preset_update, inputs=new_engine, outputs=new_preset)

        def on_book_controls(slug):
            if not slug:
                return gr.update(choices=[], value=None), "", gr.update(visible=True), ""
            choices = [("Tüm kitap / proje", 0)] + [
                (f"{chapter_id} · {title}", chapter_id) for chapter_id, title in service.chapters(slug)
            ]
            imported = bool(service.open(slug).source_format)
            return gr.update(choices=choices, value=0), service.lexicon_text(slug), gr.update(visible=not imported), ""

        project_dropdown.change(
            on_book_controls, inputs=project_dropdown, outputs=[chapter_scope, lexicon_box, save_source, lexicon_status],
        )
        chapter_scope.input(lambda: gr.update(value=None, visible=False), outputs=preview_audio)

        def on_create(name, text, document, profile_slug, upload, preset_label, lexicon_text, engine_id):
            try:
                reference = resolve_reference(profile_slug or None, upload)
                slug = service.create(
                    name, text, reference, preset_for(engine_id, preset_label), lexicon_text, engine_id=engine_id,
                    document=Path(document) if document else None,
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
            inputs=[new_name, new_text, new_document, new_profile, new_upload, new_preset, new_lexicon, new_engine],
            outputs=[create_status, project_dropdown],
        )

        def on_start(slug, chapter_id):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None)
            try:
                queued = service.start(slug, chapter_id=chapter_id or None)
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

        start_button.click(on_start, inputs=[slug_state, chapter_scope], outputs=[action_status, *view_outputs])

        def on_stop(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None)
            cancelled = service.stop(slug)
            return f"Durduruluyor: mevcut segment bitince duracak. {cancelled} segment kuyruktan çıkarıldı.", *view(slug)

        stop_button.click(on_stop, inputs=slug_state, outputs=[action_status, *view_outputs])

        def on_preview(slug, chapter_id, audio_format):
            if not slug:
                return "⚠️ Önce bir proje seçin.", gr.update(visible=False)
            audio_format = audio_format or "WAV"
            try:
                path = service.export_audio(slug, chapter_id=chapter_id or None, audio_format=audio_format.lower())
            except (StudioError, OSError) as exc:
                return f"⚠️ {exc}", gr.update(value=None, visible=False)
            return (
                f"{audio_format} hazır · `{path.name}`",
                gr.update(value=str(path), visible=True),
            )

        preview_button.click(
            on_preview, inputs=[slug_state, chapter_scope, export_format], outputs=[action_status, preview_audio],
        )

        def on_save_lexicon(slug, text, segment_id):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None), *segment_view(None, None), *take_panel(None, None)
            try:
                count = service.update_lexicon(slug, text)
                message = f"Sözlük kaydedildi. {count} segmentin okunuşu değişti."
            except StudioError as exc:
                message = f"⚠️ {exc}"
            return message, *view(slug), *segment_view(slug, segment_id), *take_panel(slug, segment_id)

        save_lexicon.click(
            on_save_lexicon, inputs=[slug_state, lexicon_box, selected_state],
            outputs=[lexicon_status, *view_outputs, *segment_outputs, *take_outputs],
        )

        def on_select(slug, ids, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not slug or row is None or row >= len(ids):
                return None, *segment_view(None, None), *take_panel(None, None)
            segment_id = ids[row]
            return segment_id, *segment_view(slug, segment_id), *take_panel(slug, segment_id)

        segment_table.select(
            on_select, inputs=[slug_state, ids_state], outputs=[selected_state, *segment_outputs, *take_outputs]
        )
        take_choice.input(take_panel, inputs=[slug_state, selected_state, take_choice], outputs=take_outputs)

        def on_take(slug, segment_id, token, action):
            if not slug or segment_id is None:
                return "⚠️ Önce bir segment seçin.", *view(slug), *take_panel(slug, segment_id), gr.skip(), gr.skip()
            try:
                item = service.segment(slug, segment_id)
                key, take_id = token.split(":") if token else (item["cache_key"], "0")
                if action == "generate":
                    service.start_take(slug, segment_id, key)
                    message = "Alternatif take kuyruğa alındı."
                elif action == "check":
                    service.check_take(slug, segment_id, key, int(take_id))
                    message = "Kalite raporu kaydedildi."
                else:
                    service.change_take(slug, segment_id, key, int(take_id) if action == "select" else None)
                    message = "Aktif take güncellendi."
            except (StudioError, ValueError) as exc:
                return f"⚠️ {exc}", *view(slug), *take_panel(slug, segment_id, token), gr.skip(), gr.skip()
            return (
                message, *view(slug), *take_panel(slug, segment_id, token if action == "check" else None),
                service.segment(slug, segment_id)["audio_path"],
                gr.update(value=None, visible=False) if action in ("select", "undo") else gr.skip(),
            )

        for button, action in (
            (take_generate, "generate"), (take_check, "check"), (take_select, "select"), (take_undo, "undo"),
        ):
            button.click(
                lambda slug, segment_id, token, action=action: on_take(slug, segment_id, token, action),
                inputs=[slug_state, selected_state, take_choice],
                outputs=[take_status, *view_outputs, *take_outputs, segment_audio, preview_audio],
            )

        def on_save(kind):
            def handler(slug, segment_id, text):
                if not slug or segment_id is None:
                    return "⚠️ Önce bir segment seçin.", *view(slug), *segment_view(slug, segment_id), *take_panel(slug, segment_id)
                try:
                    if kind == "source":
                        service.update_source(slug, segment_id, text)
                        message = "Orijinal metin kaydedildi; okunuş yeniden hesaplandı."
                    else:
                        service.set_spoken(slug, segment_id, text)
                        message = "Okunacak metin elle kaydedildi."
                except StudioError as exc:
                    return f"⚠️ {exc}", *view(slug), *segment_view(slug, segment_id), *take_panel(slug, segment_id)
                message += " Yalnız bu segment yeniden üretilecek; **Eksikleri üret** ile başlatın."
                return message, *view(slug), *segment_view(slug, segment_id), *take_panel(slug, segment_id)

            return handler

        save_source.click(
            on_save("source"),
            inputs=[slug_state, selected_state, source_box],
            outputs=[segment_status, *view_outputs, *segment_outputs, *take_outputs],
        )
        save_spoken.click(
            on_save("spoken"),
            inputs=[slug_state, selected_state, spoken_box],
            outputs=[segment_status, *view_outputs, *segment_outputs, *take_outputs],
        )

        # ---- karakterler (VF-007) ----------------------------------------
        def voice_panel(slug):
            if not slug:
                return "_Proje seçilmedi._", gr.update(choices=[(NARRATOR_LABEL, "")], value="")
            voices = service.voices(slug)
            usage: dict[str | None, int] = {}
            for item in service.segments(slug):
                usage[item["voice_id"]] = usage.get(item["voice_id"], 0) + 1
            lines = ["| Ses | Referans | Segment |", "|---|---|---:|"]
            lines.append(f"| {NARRATOR_LABEL} | proje sesi | {usage.get(None, 0)} |")
            for voice in voices:
                reference = Path(voice["reference_path"]).name if voice["reference_path"] else "—"
                lines.append(f"| {voice['name']} | `{reference}` | {usage.get(voice['key'], 0)} |")
            choices = [(NARRATOR_LABEL, "")] + [(voice["name"], voice["key"]) for voice in voices]
            return "\n".join(lines), gr.update(choices=choices, value="")

        voice_outputs = [voices_md, assign_voice]
        project_dropdown.change(voice_panel, inputs=project_dropdown, outputs=voice_outputs)
        project_dropdown.change(
            lambda slug: preset_update(service.open(slug).settings.engine_id if slug else engine_label),
            inputs=project_dropdown, outputs=voice_preset,
        )

        def on_add_voice(slug, name, profile_slug, upload, preset_label):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *voice_panel(None)
            try:
                reference = resolve_reference(profile_slug or None, upload)
                engine_id = service.open(slug).settings.engine_id
                service.add_voice(slug, name, reference, preset_for(engine_id, preset_label))
            except StudioError as exc:
                return f"⚠️ {exc}", *voice_panel(slug)
            return f"Karakter eklendi: **{name.strip()}**", *voice_panel(slug)

        add_voice_button.click(
            on_add_voice,
            inputs=[slug_state, voice_name, voice_profile, voice_upload, voice_preset],
            outputs=[voice_status, *voice_outputs],
        )

        def on_assign(slug, numbers, voice_key):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None), gr.skip()
            try:
                ids = service.segment_ids_from_numbers(slug, numbers)
                changed = service.assign_voice(slug, ids, voice_key or None)
            except StudioError as exc:
                return f"⚠️ {exc}", *view(slug), gr.skip()
            return (
                f"{changed} segmentin sesi değişti; yalnız bunlar yeniden üretilecek.",
                *view(slug),
                voice_panel(slug)[0],
            )

        assign_button.click(
            on_assign,
            inputs=[slug_state, assign_numbers, assign_voice],
            outputs=[voice_status, *view_outputs, voices_md],
        )

        def suggestions_text(slug):
            items = service.dialogue_suggestions(slug)
            if not items:
                return "_Diyalog benzeri segment bulunamadı._"
            lines = ["| # | Konuşmacı | Eşleşen karakter | Metin |", "|---:|---|---|---|"]
            for item in items:
                speaker = item["speaker"] or "belirsiz (tire/tırnak)"
                match = item["voice_key"] or ("—" if item["speaker"] else "elle atayın")
                lines.append(f"| {item['number']} | {speaker} | {match} | {_short(item['line'], 60).replace('|', '/')} |")
            return "\n".join(lines)

        def on_suggest(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin."
            return suggestions_text(slug)

        suggest_button.click(on_suggest, inputs=slug_state, outputs=suggestions_md)

        def on_apply(slug, strip):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None), gr.skip(), gr.skip()
            result = service.apply_suggestions(slug, strip_labels=bool(strip))
            message = f"{result['assigned']} segment atandı, {result['relabelled']} segmentte 'Ad:' etiketi okunuştan çıkarıldı."
            if result["unmatched"]:
                message += " Karakteri olmayan konuşmacılar: " + ", ".join(result["unmatched"]) + " (önce karakter ekleyin)."
            return message, *view(slug), suggestions_text(slug), voice_panel(slug)[0]

        apply_button.click(
            on_apply,
            inputs=[slug_state, strip_labels],
            outputs=[voice_status, *view_outputs, suggestions_md, voices_md],
        )

        def on_tick(slug, was_running, segment_id, token):
            # Yalnız üretim sürerken (ve bittiği ilk turda) yenile; boşta veritabanını yormaz.
            # Düzenleme kutularına dokunulmaz ki kullanıcının yazdığı metin ezilmesin.
            if not slug or (not was_running and not service.is_running(slug)):
                return (gr.skip(),) * (6 + len(take_outputs))
            header, _source, _spoken, _transforms, audio = segment_view(slug, segment_id)
            return *view(slug), header, audio, *take_panel(slug, segment_id, token)

        timer.tick(
            on_tick,
            inputs=[slug_state, running_state, selected_state, take_choice],
            outputs=[*view_outputs, segment_header, segment_audio, *take_outputs],
        )

    return studio
