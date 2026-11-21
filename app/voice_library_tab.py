# -*- coding: utf-8 -*-
"""Ses kütüphanesi sekmesi: kayıtlı profiller ve VoxCPM2 tasarım sesleri tek listede.

İnce Gradio katmanı; işin tamamı ``voxforge.voices.VoiceLibrary`` içinde (profil işlemleri
``scripts.voice_profile_utils``, tasarımlar ``DesignLibrary``). Bir değişiklikten sonra diğer
sekmelerdeki ses menüleri ``refresh_targets`` ile aynı olayda güncellenir.
"""

from __future__ import annotations

from typing import Any, Callable

import gradio as gr

from app.ui_theme import lead, section
from voxforge.voices import VoiceLibrary, VoiceLibraryError
from voxforge.voices.library import KIND_LABELS

TABLE_HEADERS = ["Ad", "Tür", "Kalite", "Süre (sn)", "Oluşturma"]
NO_SELECTION = "Listeden bir ses seçin."


def _quality_line(title: str, report: Any) -> str:
    if not isinstance(report, dict) or not report:
        return f"- {title}: —"
    parts = [f"**{report.get('quality') or '—'}**"]
    if report.get("duration_seconds") is not None:
        parts.append(f"{report['duration_seconds']} sn")
    if report.get("sample_rate"):
        parts.append(f"{report['sample_rate']} Hz")
    if report.get("channels"):
        parts.append(f"{report['channels']} kanal")
    line = f"- {title}: " + " · ".join(parts)
    warnings = [str(item) for item in (report.get("warnings") or [])][:3]
    return line + "".join(f"\n  - {item}" for item in warnings)


def describe_entry(item: dict[str, Any] | None) -> str:
    if not item:
        return NO_SELECTION
    lines = [f"### {item['name']}", f"{KIND_LABELS[item['kind']]} · `{item['key']}`"]
    if item.get("seconds") is not None:
        lines.append(f"Referans süresi: {item['seconds']} sn · oluşturma: {(item.get('created_at') or '—')[:10]}")
    if item["kind"] == "profile":
        lines.append(_quality_line("Yüklenen kayıt", item.get("original_quality")))
        lines.append(_quality_line("Ön işlenmiş referans", item.get("preprocessed_quality")))
        lines.append(f"- Ön işleme varyantı: `{item.get('variant') or '—'}`")
        if item.get("warning"):
            lines.append(f"- Uyarı: {item['warning']}")
    else:
        lines.append(f"- Ses tanımı: {item.get('description') or '—'}")
        lines.append(f"- Örnek replik: {item.get('text') or '—'}")
        lines.append(f"- Tohum {item.get('seed')} · model `{item.get('model_id')}`")
        lines.append("- Hızlı ses ve Studio'da referans olarak kullanılır; VoxCPM2 Studio projesinde sürüm kimliğiyle bağlanır.")
    return "\n".join(lines)


class VoiceLibraryTab:
    def __init__(self, library: VoiceLibrary) -> None:
        self.library = library
        # (dropdown, choices_fn, empty_value) — kütüphane değişince güncellenecek menüler
        self.refresh_targets: list[tuple[Any, Callable[[], list[tuple[str, str]]], Any]] = []

    # -- data -------------------------------------------------------------------------------
    def rows(self) -> list[list[Any]]:
        return [
            [item["name"], "Tasarım" if item["kind"] == "design" else "Profil", item["quality"],
             item.get("seconds") if item.get("seconds") is not None else "", (item.get("created_at") or "")[:10]]
            for item in self.library.entries()
        ]

    def broken_text(self) -> str:
        broken = self.library.broken_profiles()
        if not broken:
            return ""
        names = ", ".join(f"`{item['profile_slug']}` ({'; '.join(item.get('issues') or [])})" for item in broken)
        return f"Kullanılamayan profil klasörleri (listeye alınmadı): {names}"

    def _entry(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        try:
            return self.library.get(key)
        except VoiceLibraryError:
            return None

    def selection_outputs(self, key: str | None) -> list[Any]:
        item = self._entry(key)
        is_profile = bool(item) and item["kind"] == "profile"
        token = VoiceLibrary.confirm_token(key) if item else ""
        return [
            str(item["reference"]) if item else None,
            describe_entry(item),
            gr.update(value=item["name"] if item else ""),
            gr.update(visible=is_profile),
            gr.update(value="", info=f"Onay için `{token}` yazın." if token else "Önce bir ses seçin."),
            gr.update(interactive=bool(item)),
        ]

    def target_updates(self, current_values: list[Any]) -> list[Any]:
        updates = []
        for (component, choices_fn, empty), current in zip(self.refresh_targets, current_values):
            choices = choices_fn()
            values = {value for _label, value in choices}
            if isinstance(current, list):  # multiselect
                value = [item for item in current if item in values]
            else:
                value = current if current in values else empty
            updates.append(gr.update(choices=choices, value=value))
        return updates

    def refreshed(self, key: str | None, message: str, current_values: list[Any]) -> list[Any]:
        choices = self.library.choices()
        keys = {value for _label, value in choices}
        key = key if key in keys else None
        return [
            message,
            self.rows(),
            self.broken_text(),
            gr.update(choices=choices, value=key),
            *self.selection_outputs(key),
            *self.target_updates(current_values),
        ]

    # -- layout -----------------------------------------------------------------------------
    def build(self) -> None:
        lead(
            "Kayıtlı ses profilleri ve VoxCPM2 tasarım sesleri tek listede. Burada eklenen, yeniden adlandırılan "
            "veya silinen ses Hızlı ses ve Projeler menülerine hemen yansır."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Group(elem_classes="vf-card"):
                    section("Sesler")
                    self.table = gr.Dataframe(
                        value=self.rows(), headers=TABLE_HEADERS, datatype=["str", "str", "str", "number", "str"],
                        interactive=False, wrap=True, show_label=False, col_count=(len(TABLE_HEADERS), "fixed"),
                        column_widths=["30%", "14%", "17%", "16%", "23%"],
                    )
                    self.broken = gr.Markdown(self.broken_text(), elem_classes="vf-hint")
                    self.refresh = gr.Button("Listeyi yenile", size="sm")
                with gr.Accordion("Yeni ses profili", open=False):
                    self.new_name = gr.Textbox(label="Profil adı", placeholder="Örnek: Anlatıcı", max_lines=1)
                    self.new_audio = gr.Audio(label="Referans kayıt (6–30 sn temiz konuşma)", type="filepath")
                    self.new_permission = gr.Checkbox(label="Bu ses bana ait veya kullanma iznim var", value=False)
                    self.create = gr.Button("Profil oluştur", variant="primary")
            with gr.Column(scale=2, elem_classes="vf-rail"):
                with gr.Group(elem_classes="vf-card"):
                    section("Seçili ses")
                    self.selector = gr.Dropdown(self.library.choices(), label="Ses", value=None, filterable=False)
                    self.player = gr.Audio(label="Referans", type="filepath", interactive=False)
                    self.details = gr.Markdown(NO_SELECTION, elem_classes="vf-status")
                with gr.Accordion("Yeniden adlandır", open=False):
                    self.rename_box = gr.Textbox(label="Yeni ad", max_lines=1, info="Kimlik ve bağlantılar değişmez.")
                    self.rename_button = gr.Button("Adı kaydet", size="sm")
                with gr.Accordion("Referansı değiştir veya yeniden işle", open=False, visible=False) as self.profile_tools:
                    self.replace_audio = gr.Audio(label="Yeni referans kayıt", type="filepath")
                    self.replace_permission = gr.Checkbox(label="Bu ses bana ait veya kullanma iznim var", value=False)
                    with gr.Row():
                        self.replace_button = gr.Button("Referansı değiştir", size="sm")
                        self.reprocess_button = gr.Button("Mevcut kaydı yeniden işle", size="sm")
                    gr.Markdown(
                        "Önceki referans çöp kutusuna (`outputs/voice_library_trash`) taşınır. Studio projeleri "
                        "kendi kopyalarını kullandığı için etkilenmez.",
                        elem_classes="vf-hint",
                    )
                with gr.Accordion("Sil", open=False):
                    gr.Markdown(
                        "Ses kalıcı silinmez; `outputs/voice_library_trash` (tasarımlar Studio kökünde `trash/`) altına "
                        "taşınır. Studio projelerindeki karakter kopyaları korunur.",
                        elem_classes="vf-hint",
                    )
                    self.confirm = gr.Textbox(label="Silme onayı", max_lines=1, info="Önce bir ses seçin.")
                    self.delete_button = gr.Button("Sesi kütüphaneden kaldır", variant="stop", size="sm", interactive=False)
                self.status = gr.Markdown()

    def view_components(self) -> list[Any]:
        """Outputs of ``refreshed`` without the external menus (same order)."""
        return [self.status, self.table, self.broken, self.selector, *self.selection_components()]

    def selection_components(self) -> list[Any]:
        return [self.player, self.details, self.rename_box, self.profile_tools, self.confirm, self.delete_button]

    def wire(self) -> None:
        targets = [component for component, _fn, _empty in self.refresh_targets]
        selection = self.selection_components()
        outputs = [*self.view_components(), *targets]

        def run(action: Callable[[], tuple[str | None, str]], key: str | None, current: list[Any]) -> list[Any]:
            try:
                key, message = action()
            except VoiceLibraryError as exc:
                message = f"⚠️ {exc}"
            except Exception as exc:  # noqa: BLE001 - kullanıcıya sade gösterilir
                message = f"⚠️ İşlem tamamlanamadı: {exc}"
            return self.refreshed(key, message, current)

        def on_refresh(key, *current):
            return run(lambda: (key, ""), key, list(current))

        def on_create(name, audio, permission, key, *current):
            def action():
                slug = self.library.create_profile(name, audio, bool(permission))
                return slug, f"Profil oluşturuldu: **{self.library.get(slug)['name']}** (`{slug}`)."
            return run(action, key, list(current))

        def on_rename(key, name, *current):
            def action():
                self.library.rename(key, name)
                return key, "Ad güncellendi."
            return run(action, key, list(current))

        def on_replace(key, audio, permission, *current):
            def action():
                self.library.replace_reference(key, audio, bool(permission))
                return key, "Referans değiştirildi ve yeniden işlendi."
            return run(action, key, list(current))

        def on_reprocess(key, *current):
            def action():
                self.library.reprocess(key)
                return key, "Mevcut kayıt yeniden işlendi."
            return run(action, key, list(current))

        def on_delete(key, confirmation, *current):
            def action():
                name = self.library.get(key)["name"]
                target = self.library.delete(key, confirmation)
                return None, f"**{name}** kütüphaneden kaldırıldı; dosyalar `{target.name}` adıyla çöp kutusunda."
            return run(action, key, list(current))

        def on_table_select(evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            entries = self.library.entries()
            return gr.update(value=entries[row]["key"]) if row is not None and 0 <= row < len(entries) else gr.update()

        self.refresh.click(on_refresh, [self.selector, *targets], outputs)
        def clear_form(prefix: str):
            def clear(status):
                done = str(status or "").startswith(prefix)
                return [gr.update(value=None) if done else gr.update() for _ in range(2)] + [gr.update(value=False) if done else gr.update()]
            return clear

        self.create.click(on_create, [self.new_name, self.new_audio, self.new_permission, self.selector, *targets], outputs).then(
            clear_form("Profil oluşturuldu"), self.status, [self.new_name, self.new_audio, self.new_permission],
        )
        self.rename_button.click(on_rename, [self.selector, self.rename_box, *targets], outputs)
        self.replace_button.click(on_replace, [self.selector, self.replace_audio, self.replace_permission, *targets], outputs).then(
            lambda status: [gr.update(value=None), gr.update(value=False)] if str(status or "").startswith("Referans değiştirildi")
            else [gr.update(), gr.update()],
            self.status, [self.replace_audio, self.replace_permission],
        )
        self.reprocess_button.click(on_reprocess, [self.selector, *targets], outputs)
        self.delete_button.click(on_delete, [self.selector, self.confirm, *targets], outputs)
        self.selector.change(self.selection_outputs, self.selector, selection)
        self.table.select(on_table_select, None, self.selector)
