# -*- coding: utf-8 -*-
"""VoxForge Voice Playground: free Turkish text with XTTS-v2, Chatterbox V3 or VoxCPM2.

Standalone Gradio app (127.0.0.1:7861 by default), separate from the Studio demo.
Run with run_voice_playground.ps1. Only one engine is on the GPU at a time.
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from voxforge.playground.service import (  # noqa: E402
    DEFAULT_SEED,
    ENGINE_LABELS,
    MAX_TEXT_CHARS,
    VARIANTS,
    PlaygroundService,
    describe,
    list_profiles,
)

SLOTS = 4
ENGINE_CHOICES = [(label, engine) for engine, label in ENGINE_LABELS.items()]
CHATTERBOX_CHOICES = [("Varsayılan", "chatterbox-default"), ("min_p = 0.02", "chatterbox-minp")]
COMPARE_CHOICES = [(v.label, v.key) for v in VARIANTS.values()]
SAMPLE_TEXT = "Merhaba! Bugün 5 Kasım 2026; hava 18 derece. Evet, tamam, birazdan başlıyoruz."


def variant_for(engine: str, chatterbox_setting: str) -> str:
    if engine == "chatterbox-v3":
        return chatterbox_setting or "chatterbox-default"
    return next(v.key for v in VARIANTS.values() if v.engine == engine)


def profile_choices() -> list[tuple[str, str]]:
    return [(f"{p['name']} ({p['slug']})", p["slug"]) for p in list_profiles()]


def history_choices(service: PlaygroundService) -> list[tuple[str, str]]:
    choices = []
    for record in service.history():
        text = record.get("text", "").replace("\n", " ")
        short = text if len(text) <= 40 else text[:37] + "..."
        choices.append((f"{record['created_at'][11:19]} · {record['label']} · {record.get('profile') or '-'} · {short}", record["id"]))
    return choices


def status_line(service: PlaygroundService) -> str:
    status = service.manager.status()
    if not status.get("engine"):
        return "GPU'da yüklü motor yok."
    return (
        f"Yüklü motor: **{ENGINE_LABELS.get(status['engine'], status['engine'])}** "
        f"(cihaz {status.get('device')}, yükleme {status.get('load_seconds')} sn, PID {status.get('pid')}). "
        "Başka motor seçilince bu süreç kapatılır."
    )


def slot_updates(service: PlaygroundService, ids: list[str]) -> list[Any]:
    """Audio + Markdown pairs for the comparison slots."""
    updates: list[Any] = []
    records = [r for r in (service.get(i) for i in ids[:SLOTS]) if r]
    for index in range(SLOTS):
        if index < len(records):
            record = records[index]
            updates += [gr.update(value=record["wav"], label=record["label"], visible=True), gr.update(value=describe(record), visible=True)]
        else:
            updates += [gr.update(value=None, visible=False), gr.update(value="", visible=False)]
    return updates


def build_app(service: PlaygroundService | None = None) -> gr.Blocks:
    service = service or PlaygroundService()

    with gr.Blocks(title="VoxForge Voice Playground") as app:
        gr.Markdown(
            "# VoxForge Voice Playground\n"
            "Türkçe metni seçtiğin motor ve kayıtlı ses profiliyle üret, dinle, indir ve karşılaştır. "
            "GPU'da aynı anda tek motor çalışır; motor değişince önceki kapatılır (ilk üretimde model yüklemesi sürer)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label=f"Metin (en fazla {MAX_TEXT_CHARS} karakter)", value=SAMPLE_TEXT, lines=5, max_lines=14)
                with gr.Row():
                    engine = gr.Radio(ENGINE_CHOICES, value="xtts", label="Motor")
                    chatterbox_setting = gr.Radio(CHATTERBOX_CHOICES, value="chatterbox-default", label="Chatterbox ayarı", visible=False)
                with gr.Row():
                    profile = gr.Dropdown(profile_choices(), label="Ses profili", value=None, allow_custom_value=False)
                    refresh_profiles = gr.Button("Profilleri yenile", size="sm")
                with gr.Row():
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Tohum (-1 = rastgele)")
                    normalize = gr.Checkbox(value=True, label="Türkçe okunuş normalizasyonu (sayı, tarih, saat)")
                generate = gr.Button("Üret", variant="primary")
            with gr.Column(scale=2):
                result_audio = gr.Audio(label="Son üretim", type="filepath", interactive=False)
                result_file = gr.File(label="WAV indir", interactive=False)
                result_info = gr.Markdown()
                status = gr.Markdown(status_line(service))
                unload = gr.Button("Motoru kapat (VRAM boşalt)", size="sm")

        with gr.Accordion("Toplu karşılaştırma: aynı metin, aynı tohum, birden çok motor/ayar", open=False):
            compare_variants = gr.CheckboxGroup(
                COMPARE_CHOICES,
                value=["xtts", "chatterbox-default", "chatterbox-minp"],
                label="Varyantlar (motor sırasına göre üretilir, her model bir kez yüklenir)",
            )
            generate_many = gr.Button("Seçilenlerin hepsini üret")

        gr.Markdown("## Yan yana dinle")
        selected = gr.Dropdown(
            history_choices(service),
            value=[r["id"] for r in service.history()[:SLOTS]],
            multiselect=True,
            max_choices=SLOTS,
            label=f"Karşılaştırılacak üretimler (en fazla {SLOTS}; yeni üretimler başa eklenir)",
        )
        slot_components: list[Any] = []
        with gr.Row(equal_height=False):
            for _ in range(SLOTS):
                with gr.Column(min_width=240):
                    audio = gr.Audio(type="filepath", interactive=False, visible=False)
                    info = gr.Markdown(visible=False)
                    slot_components += [audio, info]
        message = gr.Markdown()

        # -- handlers ------------------------------------------------------------------------
        def on_engine(value: str) -> Any:
            return gr.update(visible=value == "chatterbox-v3")

        def after_records(records: list[dict[str, Any]], current: list[str] | None, errors: list[str]) -> list[Any]:
            new_ids = [r["id"] for r in records]
            ids = list(dict.fromkeys(new_ids + list(current or [])))[:SLOTS]
            last = records[-1] if records else None
            text_lines = [f"Hata — {e}" for e in errors]
            return [
                last["wav"] if last else gr.update(),
                last["wav"] if last else gr.update(),
                describe(last) if last else gr.update(),
                status_line(service),
                gr.update(choices=history_choices(service), value=ids),
                "\n\n".join(text_lines),
                *slot_updates(service, ids),
            ]

        def on_generate(text_value, engine_value, chatter_value, profile_value, seed_value, normalize_value, current):
            key = variant_for(engine_value, chatter_value)
            records, errors = service.generate_many(text_value, [key], profile_value, seed=_seed(seed_value), normalize=normalize_value)
            return after_records(records, current, errors)

        def on_generate_many(text_value, keys, profile_value, seed_value, normalize_value, current):
            if not keys:
                return after_records([], current, ["En az bir varyant seçin."])
            records, errors = service.generate_many(text_value, list(keys), profile_value, seed=_seed(seed_value), normalize=normalize_value)
            return after_records(records, current, errors)

        def on_unload():
            engine_name = service.manager.unload()
            note = f"{ENGINE_LABELS.get(engine_name, engine_name)} kapatıldı." if engine_name else "Yüklü motor yoktu."
            return status_line(service), note

        outputs = [result_audio, result_file, result_info, status, selected, message, *slot_components]
        engine.change(on_engine, engine, chatterbox_setting)
        refresh_profiles.click(lambda: gr.update(choices=profile_choices()), None, profile)
        generate.click(on_generate, [text, engine, chatterbox_setting, profile, seed, normalize, selected], outputs)
        generate_many.click(on_generate_many, [text, compare_variants, profile, seed, normalize, selected], outputs)
        selected.change(lambda ids: slot_updates(service, list(ids or [])), selected, slot_components)
        unload.click(on_unload, None, [status, message])
        app.load(lambda ids: slot_updates(service, list(ids or [])), selected, slot_components)

    app.queue(default_concurrency_limit=1)
    return app


def _seed(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    service = PlaygroundService()
    atexit.register(service.manager.close)
    service.audio_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(service)
    port = int(os.environ.get("VOXFORGE_PLAYGROUND_PORT", "7861"))
    print(f"Voice Playground çıktıları: {service.root}")
    # outputs/ bir junction olabilir; Gradio çözülmüş yolu denetler.
    app.launch(server_name="127.0.0.1", server_port=port, share=False, allowed_paths=[str(service.root.resolve())])


if __name__ == "__main__":
    main()
