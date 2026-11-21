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

from app.ui_theme import section
from voxforge.quality.checks import describe_report, summarize
from voxforge.studio.service import NARRATOR_LABEL, STATE_LABELS, StudioError, StudioService

TABLE_HEADERS = ["Parça", "Metin", "Karakter", "Durum", "Süre (sn)"]
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

    def design_choices() -> list[tuple[str, str]]:
        return [
            (f"{item['name']} · {item['created_at'][:10]} · {item['id'][:8]}", item["id"])
            for item in service.designs.versions()
        ]

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
                    _short(item["source_text"]),
                    item["voice_name"] or NARRATOR_LABEL,
                    STATE_LABELS[item["state"]],
                    round(item["duration_s"], 2) if item["duration_s"] is not None else None,
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
            f"**Parça {item['ord'] + 1}** · {item['chapter_title'] or 'Metin'} · {STATE_LABELS[item['state']]}"
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

    def mix_panel(slug, segment_id):
        if not slug or segment_id is None:
            return (
                gr.update(value=0, interactive=False), gr.update(value=None, interactive=False),
                gr.update(interactive=False), gr.update(interactive=False),
            )
        item = service.segment(slug, segment_id)
        return (
            gr.update(value=item["gain_db"], interactive=True),
            gr.update(value=item["pause_s"], interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=item["can_undo_mix"]),
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
        if not vox_presets:
            gr.Markdown("VoxCPM2 üretimi için Voice Playground uygulamasında **Projeler → Sesli kitap** bölümünü açın.", elem_classes="vf-hint")
        slug_state = gr.State(None)
        ids_state = gr.State([])
        selected_state = gr.State(None)
        running_state = gr.State(False)

        with gr.Group(elem_classes="vf-card"):
            section("Sesli kitap")
            with gr.Row(equal_height=True):
                project_dropdown = gr.Dropdown(label="Kayıtlı kitabı aç", choices=project_choices(), value=None, scale=4)
                refresh_projects = gr.Button("Listeyi yenile", size="sm", scale=1, min_width=120)

        with gr.Tabs(elem_classes="vf-steps") as book_steps:
            with gr.Tab("1. Kitap", id="book"):
                with gr.Group(elem_classes="vf-card"):
                    section("Yeni kitap", "Bir metin yapıştır veya kitap dosyanı yükle.")
                    new_name = gr.Textbox(label="Proje adı")
                    new_text = gr.Textbox(label="Kitap metni", lines=4, placeholder="Seslendirilecek Türkçe metni yapıştırın.")
                    new_document = gr.File(
                        label="veya kitap dosyası (TXT, DOCX, EPUB)", type="filepath", file_types=[".txt", ".docx", ".epub"],
                    )
                    gr.Markdown(
                        "TXT’de `Bölüm 1: Başlık` veya `# Başlık` kullanın. Paragrafları boş satırla ayırın.",
                        elem_classes="vf-hint",
                    )
                    with gr.Accordion("Anlatıcı sesi ve okuma ayarları", open=False):
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
                            "Seçilen referans proje klasörüne kopyalanır; profil sonradan değişse de proje etkilenmez.",
                            elem_classes="vf-hint",
                        )
                    create_button = gr.Button("Projeyi oluştur", variant="primary")
                    create_status = gr.Markdown()
                    to_voices = gr.Button("Karakter seslerini seç", visible=False)

                with gr.Accordion("Proje yedeği", open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            backup_button = gr.Button("Seçili projeyi ZIP olarak yedekle")
                            backup_file = gr.File(label="Proje yedeği", interactive=False, height=72, elem_classes="vf-file")
                        with gr.Column():
                            restore_file = gr.File(label="Geri yüklenecek VoxForge ZIP", type="filepath", file_types=[".zip"])
                            restore_name = gr.Textbox(label="Geri yüklenen projenin adı")
                            restore_button = gr.Button("Yeni proje olarak geri yükle")
                    backup_status = gr.Markdown()

            with gr.Tab("2. Sesler", id="voices"):
                with gr.Group(elem_classes="vf-card"):
                    section("Kim konuşuyor?")
                    gr.Markdown(
                        "Kitabın tamamını anlatıcı okuyacaksa doğrudan **Üretime geç**. "
                        "Diyaloglar için önce karakteri ekle, sonra okuyacağı parçaları seç."
                    )
                    voices_md = gr.Markdown()
                    with gr.Row():
                        voice_name = gr.Textbox(label="Karakter adı", scale=2)
                        voice_profile = gr.Dropdown(label="Klonlama için ses profili", choices=profile_choices(), value="", scale=2)
                        voice_upload = gr.Audio(label="veya klonlama için referans ses", type="filepath", scale=2)
                        voice_preset = gr.Dropdown(label="Üretim modu", choices=preset_choices, value=default_preset, scale=1)
                    add_voice_button = gr.Button("Karakter ekle")
                    with gr.Row():
                        assign_numbers = gr.Dropdown(label="Ses verilecek parçalar", choices=[], multiselect=True, scale=3)
                        assign_voice = gr.Dropdown(label="Ses", choices=[(NARRATOR_LABEL, "")], value="", scale=2)
                        assign_button = gr.Button("Sesi parçalara ata", scale=1)
                    with gr.Row():
                        suggest_button = gr.Button("Diyalog önerilerini göster")
                        strip_labels = gr.Checkbox(label="'Ad:' etiketini okunuştan çıkar", value=True)
                        apply_button = gr.Button("Eşleşen önerileri uygula")
                    suggestions_md = gr.Markdown()
                    voice_status = gr.Markdown()

                if vox_presets:
                    with gr.Accordion("Ses tasarımı (VoxCPM2)", open=False):
                        gr.Markdown(
                            "Ses tanımından referanssız bir örnek üretin. Studio, tasarımın ses kimliğini "
                            "sürdürmek için kaydedilen örneği karakter sesi olarak kullanır."
                        )
                        design_token = gr.State(None)
                        design_description = gr.Textbox(label="Ses tanımı", placeholder="Yetişkin, sıcak, sakin ve tok bir ses")
                        design_text = gr.Textbox(label="Önizleme repliği", lines=2)
                        design_seed = gr.Number(label="Tohum", value=1234, precision=0)
                        design_preview_button = gr.Button("Referanssız örnek üret", variant="primary")
                        design_audio = gr.Audio(label="Tasarım önizlemesi", type="filepath", interactive=False)
                        with gr.Row():
                            design_name = gr.Textbox(label="Kütüphane adı")
                            design_save = gr.Button("Yeni sürüm kaydet", interactive=False)
                        design_choice = gr.Dropdown(
                            label="Kaydedilmiş tasarım sürümü", choices=design_choices(),
                        )
                        saved_design_audio = gr.Audio(label="Kaydedilmiş örnek", type="filepath", interactive=False)
                        design_character = gr.Textbox(label="Studio karakter adı (boşsa kütüphane adı)")
                        design_add = gr.Button("Tasarımı projeye karakter olarak ekle")
                        design_status = gr.Markdown()

                to_production = gr.Button("Üretime geç", variant="primary")
            with gr.Tab("3. Üretim", id="production"):
                progress_md = gr.Markdown("Proje seçilmedi.", elem_classes="vf-progress")
                with gr.Group(elem_classes="vf-card"):
                    section(
                        "Üret ve dışa aktar",
                        "Kapsam tüm kitap ya da tek bölüm olabilir. Yalnız eksik, değişen ve hatalı segmentler üretilir.",
                    )
                    with gr.Row(equal_height=True):
                        chapter_scope = gr.Dropdown(label="Üretim ve dışa aktarma kapsamı", choices=[], scale=3)
                        start_button = gr.Button(
                            "Eksikleri ve hatalıları üret / devam et", variant="primary", scale=2, elem_classes="vf-go",
                        )
                        stop_button = gr.Button("Durdur", variant="stop", scale=1)
                    with gr.Row(equal_height=True):
                        export_format = gr.Dropdown(label="Ses biçimi", choices=["WAV", "MP3"], value="WAV", scale=3)
                        preview_button = gr.Button("Dışa aktar", scale=3)
                action_status = gr.Markdown()
                preview_audio = gr.Audio(label="Ses çıktısı", type="filepath", interactive=False, visible=False)

                with gr.Accordion("Proje telaffuz sözlüğü", open=False):
                    lexicon_box = gr.Textbox(label="Girişler", lines=5, placeholder="! Özel İsim = okunuş")
                    save_lexicon = gr.Button("Sözlüğü kaydet")
                    lexicon_status = gr.Markdown()

                segment_table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["number", "str", "str", "str", "number"],
                    interactive=False,
                    wrap=True,
                    label="Kitabın parçaları · dinlemek veya düzenlemek için bir satır seç",
                    max_height=520,
                )

                with gr.Row(equal_height=False, visible=False) as part_editor:
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes="vf-card"):
                            segment_header = gr.Markdown(NO_SELECTION, elem_classes="vf-section")
                            segment_audio = gr.Audio(label="Aktif take · kaynak WAV", type="filepath", interactive=False)
                            source_box = gr.Textbox(label="Orijinal metin", lines=3)
                            save_source = gr.Button("Orijinali kaydet (okunuş yeniden hesaplanır)", size="sm")
                            spoken_box = gr.Textbox(label="Okunacak metin", lines=3)
                            save_spoken = gr.Button("Okunuşu elle kaydet", size="sm")
                            transformations_md = gr.Markdown()
                            segment_status = gr.Markdown()
                    with gr.Column(scale=2):
                        with gr.Accordion("Dışa aktarma sesi", open=False):
                            gain_slider = gr.Slider(-24, 12, value=0, step=0.5, label="Dışa aktarma seviyesi (dB)", interactive=False)
                            pause_box = gr.Number(
                                label="Sonraki duraklama (sn)", minimum=0, maximum=5, interactive=False,
                                info="Boşsa üretim modu kullanılır. Dışa aktarılan bölümde veya kitapta bir sonraki segment varsa uygulanır.",
                            )
                            with gr.Row():
                                save_mix = gr.Button("Ses ayarını kaydet", interactive=False, size="sm")
                                undo_mix = gr.Button("Ses ayarını geri al", interactive=False, size="sm")
                            mix_status = gr.Markdown()
                        with gr.Group(elem_classes="vf-card"):
                            section("Segment kalitesi ve take'ler")
                            take_choice = gr.Dropdown(label="Take", choices=[])
                            take_quality = gr.Markdown()
                            take_audio = gr.Audio(label="Alternatif take sesi", type="filepath", interactive=False)
                            segment_generate = gr.Button("Eksik veya değişen bu segmenti üret", variant="primary", size="sm")
                            take_generate = gr.Button(
                                "Bu segmenti alternatif take olarak üret", visible=bool(vox_presets), size="sm",
                            )
                            with gr.Row():
                                take_check = gr.Button("Kaliteyi denetle", size="sm")
                                take_select = gr.Button("Take'i kullan", interactive=False, size="sm")
                                take_undo = gr.Button("Take seçimini geri al", interactive=False, size="sm")
                            take_status = gr.Markdown()

        project_dropdown.change(lambda slug: gr.update(visible=bool(slug)), project_dropdown, to_voices)
        to_voices.click(lambda: gr.update(selected="voices"), outputs=book_steps)
        to_production.click(lambda: gr.update(selected="production"), outputs=book_steps)
        selected_state.change(lambda part: gr.update(visible=part is not None), selected_state, part_editor)

        timer = gr.Timer(REFRESH_SECONDS)

        view_outputs = [progress_md, segment_table, ids_state, running_state]
        segment_outputs = [segment_header, source_box, spoken_box, transformations_md, segment_audio]
        take_outputs = [take_choice, take_quality, take_audio, take_select, take_undo]
        mix_outputs = [gain_slider, pause_box, save_mix, undo_mix]

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
            lambda: (*take_panel(None, None), *mix_panel(None, None), "", gr.update(value=None, visible=False)),
            outputs=[*take_outputs, *mix_outputs, mix_status, preview_audio],
        )
        refresh_projects.click(lambda: gr.update(choices=project_choices()), outputs=project_dropdown)
        new_engine.change(preset_update, inputs=new_engine, outputs=new_preset)

        def on_backup(slug):
            if not slug:
                return "⚠️ Önce bir proje seçin.", None
            try:
                return "Proje yedeği hazır.", str(service.backup(slug))
            except StudioError as exc:
                return f"⚠️ {exc}", None

        def on_restore(bundle, name):
            if not bundle:
                return "⚠️ Bir VoxForge ZIP yedeği seçin.", gr.skip(), gr.skip()
            try:
                slug = service.restore(Path(bundle), name)
                return "Yedek yeni proje olarak açıldı.", gr.update(choices=project_choices(), value=slug), None
            except StudioError as exc:
                return f"⚠️ {exc}", gr.skip(), gr.skip()

        backup_button.click(on_backup, inputs=slug_state, outputs=[backup_status, backup_file])
        restore_button.click(
            on_restore, inputs=[restore_file, restore_name], outputs=[backup_status, project_dropdown, restore_file],
        )

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
                f"**{name}** hazır ({count} segment). Karakterlere ses vermek için **2. Sesler** adımını aç.",
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
                path, warnings = service.export_audio_with_report(
                    slug, chapter_id=chapter_id or None, audio_format=audio_format.lower(),
                )
            except (StudioError, OSError) as exc:
                return f"⚠️ {exc}", gr.update(value=None, visible=False)
            status = f"{audio_format} hazır · `{path.name}`"
            if warnings:
                status += "\n\n" + "\n".join(f"- {warning}" for warning in warnings)
            return status, gr.update(value=str(path), visible=True)

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
            if not slug or row is None or not 0 <= row < len(ids):
                return None, *segment_view(None, None), *take_panel(None, None), *mix_panel(None, None), ""
            segment_id = ids[row]
            return segment_id, *segment_view(slug, segment_id), *take_panel(slug, segment_id), *mix_panel(slug, segment_id), ""

        segment_table.select(
            on_select, inputs=[slug_state, ids_state],
            outputs=[selected_state, *segment_outputs, *take_outputs, *mix_outputs, mix_status],
        )

        def on_mix(slug, segment_id, mix):
            if not slug or segment_id is None:
                return "⚠️ Önce bir segment seçin.", *view(slug), *mix_panel(None, None), gr.skip()
            try:
                service.change_mix(slug, segment_id, mix)
            except StudioError as exc:
                return f"⚠️ {exc}", *view(slug), *mix_panel(slug, segment_id), gr.skip()
            message = "Son ses düzenlemesi geri alındı." if mix is None else "Ses ayarı kaydedildi."
            return message, *view(slug), *mix_panel(slug, segment_id), gr.update(value=None, visible=False)

        save_mix.click(
            lambda slug, segment_id, gain, pause: on_mix(slug, segment_id, (gain, pause)),
            inputs=[slug_state, selected_state, gain_slider, pause_box],
            outputs=[mix_status, *view_outputs, *mix_outputs, preview_audio],
        )
        undo_mix.click(
            lambda slug, segment_id: on_mix(slug, segment_id, None),
            inputs=[slug_state, selected_state], outputs=[mix_status, *view_outputs, *mix_outputs, preview_audio],
        )
        take_choice.input(take_panel, inputs=[slug_state, selected_state, take_choice], outputs=take_outputs)

        def on_take(slug, segment_id, token, action):
            if not slug or segment_id is None:
                return "⚠️ Önce bir segment seçin.", *view(slug), *take_panel(slug, segment_id), gr.skip(), gr.skip()
            try:
                item = service.segment(slug, segment_id)
                key, take_id = token.split(":") if token else (item["cache_key"], "0")
                if action == "segment":
                    queued = service.start_segment(slug, segment_id)
                    message = "Seçili segment kuyruğa alındı." if queued else "Seçili segment için yeni iş başlatılmadı."
                elif action == "generate":
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
            (segment_generate, "segment"), (take_generate, "generate"), (take_check, "check"),
            (take_select, "select"), (take_undo, "undo"),
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
                message += " Bu repliği tek başına üretmek için **Eksik veya değişen bu segmenti üret** düğmesini kullanın."
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
            lines = ["| Ses | Ses kaynağı | Segment |", "|---|---|---:|"]
            lines.append(f"| {NARRATOR_LABEL} | proje sesi | {usage.get(None, 0)} |")
            for voice in voices:
                reference = Path(voice["reference_path"]).name if voice["reference_path"] else "—"
                if voice["design_version"]:
                    reference = f"Tasarlanmış ses · {voice['design_version'][:8]}"
                elif voice["reference_path"]:
                    reference = f"Referanslı klonlama · `{reference}`"
                lines.append(f"| {voice['name']} | {reference} | {usage.get(voice['key'], 0)} |")
            choices = [(NARRATOR_LABEL, "")] + [(voice["name"], voice["key"]) for voice in voices]
            return "\n".join(lines), gr.update(choices=choices, value="")

        voice_outputs = [voices_md, assign_voice]
        def assignment_choices(slug):
            parts = service.segments(slug) if slug else []
            choices = [(f"{part['ord'] + 1} · {_short(part['source_text'], 55)}", str(part["ord"] + 1)) for part in parts]
            return gr.update(choices=choices, value=[])

        project_dropdown.change(assignment_choices, project_dropdown, assign_numbers)
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

        if vox_presets:
            def on_design_preview(description, text, seed):
                try:
                    sample = service.preview_design(description, text, seed)
                except StudioError as exc:
                    return f"⚠️ {exc}", None, None, gr.update(interactive=False)
                return "Referanssız örnek hazır.", str(sample["wav"]), sample["id"], gr.update(interactive=True)

            design_preview_button.click(
                on_design_preview, inputs=[design_description, design_text, design_seed],
                outputs=[design_status, design_audio, design_token, design_save],
            )

            def on_save_design(token, name):
                if not token:
                    return "⚠️ Önce bir örnek üretin.", gr.skip()
                try:
                    version = service.save_design(token, name or "")
                except StudioError as exc:
                    return f"⚠️ {exc}", gr.skip()
                return "Yeni sürüm kaydedildi.", gr.update(choices=design_choices(), value=version["id"])

            design_save.click(on_save_design, inputs=[design_token, design_name], outputs=[design_status, design_choice])

            def on_design_version(token):
                if not token:
                    return None
                try:
                    return str(service.designs.version(token)["wav"])
                except (ValueError, OSError, KeyError) as exc:
                    raise gr.Error(f"Ses tasarımı açılamadı: {exc}") from exc

            design_choice.change(on_design_version, inputs=design_choice, outputs=saved_design_audio)

            def on_add_design(slug, token, name, preset_label):
                if not slug or not token:
                    return "⚠️ Bir VoxCPM2 projesi ve tasarım sürümü seçin.", *voice_panel(slug)
                try:
                    service.add_designed_voice(slug, name or "", token, preset_for("voxcpm2", preset_label))
                except StudioError as exc:
                    return f"⚠️ {exc}", *voice_panel(slug)
                return "Tasarlanmış ses eklendi. Karakterler bölümünden segmentlere atayın.", *voice_panel(slug)

            design_add.click(
                on_add_design, inputs=[slug_state, design_choice, design_character, voice_preset],
                outputs=[design_status, *voice_outputs],
            )

        def on_assign(slug, numbers, voice_key):
            if not slug:
                return "⚠️ Önce bir proje seçin.", *view(None), gr.skip()
            try:
                ids = service.segment_ids_from_numbers(slug, ", ".join(numbers or []))
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
