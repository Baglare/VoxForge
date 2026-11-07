# -*- coding: utf-8 -*-
"""VoxForge Voice Playground: free Turkish text with XTTS-v2, Chatterbox V3 or VoxCPM2.

Standalone Gradio app (127.0.0.1:7861 by default), separate from the Studio demo.
Run with run_voice_playground.ps1. Only one engine is on the GPU at a time.

VF-016: VoxCPM2 advanced controls (CFG, diffusion steps, seed, style instruction kept
apart from the text, narration pacing), built-in and user presets, preset comparison and
a "Sahne anlatımı" tab (scene-based narration merged into one WAV).
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

from voxforge.playground.prosody import (  # noqa: E402
    CFG_RANGE,
    PACINGS,
    PAUSE_RANGE,
    STEPS_RANGE,
    ULTIMATE_COMPATIBILITY,
    ControlError,
    VoxControls,
)
from voxforge.playground.scenes import SCENE_KINDS, SceneNarrator, SceneProject, slugify, split_scenes  # noqa: E402
from voxforge.playground.service import (  # noqa: E402
    CONTROLLABLE_VARIANTS,
    DEFAULT_SEED,
    DEVICE_LABELS,
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
VOXCPM_CHOICES = [
    ("Varsayılan (gelişmiş kontrollerle)", "voxcpm2"),
    ("Hızlı (6 adım, sabit deneme)", "voxcpm2-fast"),
    ("Ultimate Cloning (referans + transkript)", "voxcpm2-ultimate"),
]
DEVICE_CHOICES = [(label, key) for key, label in DEVICE_LABELS.items()]
COMPARE_CHOICES = [(v.label, v.key) for v in VARIANTS.values()]
PACING_CHOICES = [(label, key) for key, label in PACINGS.items()]
KIND_CHOICES = [(label, kind) for kind, (label, _preset) in SCENE_KINDS.items()]
SCENE_HEADERS = ["#", "Başlık", "Tür", "Preset", "Metin", "Durum"]
SAMPLE_TEXT = "Merhaba! Bugün 5 Kasım 2026; hava 18 derece. Evet, tamam, birazdan başlıyoruz."
SAMPLE_SCENES = (
    "## Kale\nKrallığın kuzeyinde, yüzyıllardır kimsenin yaklaşmaya cesaret edemediği eski bir kale vardı.\n\n"
    "## Gece\nFakat o gece, nöbetçiler içeriden gelen ayak seslerini duydu. Hiç kimse orada yaşamıyordu.\n\n"
    "## Saldırı\nBirden kapılar açıldı ve düşman askerleri kılıçlarıyla avluya hücum etti!\n\n"
    "## Veda\nYaşlı nöbetçi, gözyaşlarını saklamadan oğluna son kez sarıldı."
)
ULTIMATE_NOTE = "Ultimate Cloning ile uyumluluk: " + " · ".join(f"**{k}**: {v}" for k, v in ULTIMATE_COMPATIBILITY.items())


def variant_for(engine: str, chatterbox_setting: str, voxcpm_setting: str = "voxcpm2") -> str:
    if engine == "chatterbox-v3":
        return chatterbox_setting or "chatterbox-default"
    if engine == "voxcpm2":
        return voxcpm_setting or "voxcpm2"
    return next(v.key for v in VARIANTS.values() if v.engine == engine)


def profile_choices() -> list[tuple[str, str]]:
    return [(f"{p['name']} ({p['slug']})", p["slug"]) for p in list_profiles()]


def preset_choices(service: PlaygroundService) -> list[tuple[str, str]]:
    return [(("" if p.builtin else "★ ") + p.name, key) for key, p in service.presets.all().items()]


def preset_info(service: PlaygroundService, key: str | None) -> str:
    if not key:
        return ""
    try:
        preset = service.presets.get(key)
    except ControlError as exc:
        return str(exc)
    return f"**{preset.name}** — {preset.controls.label()}  \nKanıt: {preset.evidence}"


def history_choices(service: PlaygroundService) -> list[tuple[str, str]]:
    choices = []
    for record in service.history():
        text = record.get("text", "").replace("\n", " ")
        short = text if len(text) <= 40 else text[:37] + "..."
        preset = (record.get("settings") or {}).get("preset")
        label = record["label"] + (f" [{preset}]" if preset else "")
        choices.append((f"{record['created_at'][11:19]} · {label} · {record.get('profile') or '-'} · {short}", record["id"]))
    return choices


def _mb(value: Any) -> str:
    return f"{value:,.0f} MB".replace(",", ".") if isinstance(value, (int, float)) else "—"


def status_line(service: PlaygroundService) -> str:
    status = service.manager.status()
    if not status.get("engine"):
        return "Yüklü motor yok (GPU boş)."
    runtime = status.get("runtime") or {}
    vram = status.get("vram") or {}
    device = status.get("device")
    lines = [
        f"Yüklü motor: **{ENGINE_LABELS.get(status['engine'], status['engine'])}** · cihaz **{DEVICE_LABELS.get(device, device)}**"
        f" · yükleme {status.get('load_seconds')} sn · PID {status.get('pid')}",
    ]
    if runtime.get("gpu"):
        lines.append(
            f"GPU: {runtime['gpu']} · torch {runtime.get('torch')} · VRAM (yüklemeden sonra) {_mb(vram.get('device_used_mb'))}"
            f"/{_mb(vram.get('device_total_mb'))}"
        )
    elif status["engine"] != "mock":
        lines.append(f"CPU: torch {runtime.get('torch')} · {runtime.get('cpu_threads')} iş parçacığı (GPU kullanılmıyor)")
    extras = [f"dtype {status['dtype']}" if status.get("dtype") else "", status.get("compile") and f"torch.compile {status['compile']}"]
    extras = [item for item in extras if item]
    if extras:
        lines.append(" · ".join(extras))
    lines.append("Başka motor veya cihaz seçilince bu süreç kapatılır.")
    return "  \n".join(lines)


def slot_updates(service: PlaygroundService, ids: list[str]) -> list[Any]:
    """Audio + Markdown pairs for the comparison slots."""
    updates: list[Any] = []
    records = [r for r in (service.get(i) for i in ids[:SLOTS]) if r]
    for index in range(SLOTS):
        if index < len(records):
            record = records[index]
            label = record["label"] + (f" · {record['settings']['preset']}" if (record.get("settings") or {}).get("preset") else "")
            updates += [gr.update(value=record["wav"], label=label, visible=True), gr.update(value=describe(record), visible=True)]
        else:
            updates += [gr.update(value=None, visible=False), gr.update(value="", visible=False)]
    return updates


def controls_from(cfg: Any, steps: Any, style: Any, pacing: Any, sentence_pause: Any, paragraph_pause: Any) -> VoxControls:
    return VoxControls(
        cfg=float(cfg), steps=int(steps), style=style or "", pacing=pacing or "standard",
        sentence_pause=float(sentence_pause), paragraph_pause=float(paragraph_pause),
    ).validated()


SCENE_STATE_LABELS = {"missing": "üretilmedi", "stale": "ayar değişti, yeniden üret"}


def scene_rows(project: SceneProject | None, narrator: SceneNarrator | None = None, device: str = "cuda") -> list[list[Any]]:
    if project is None:
        return []
    rows = []
    for index, scene in enumerate(project.scenes, start=1):
        state_key = narrator.scene_state(project, scene, device) if narrator else ("missing" if not scene.wav else "stale")
        state = SCENE_STATE_LABELS.get(state_key) or f"hazır ({scene.audio_seconds} sn)"
        rows.append([index, scene.title, scene.kind, scene.preset, scene.text, state])
    return rows


def apply_rows(project: SceneProject, rows: Any, presets: dict[str, Any]) -> SceneProject:
    """Copy the user's table edits (title, kind, preset, text) back into the project."""
    values = rows.values.tolist() if hasattr(rows, "values") else list(rows or [])
    for scene, row in zip(project.scenes, values):
        title, kind, preset, text = (str(row[1] or "").strip(), str(row[2] or "").strip(), str(row[3] or "").strip(), str(row[4] or "").strip())
        if kind not in SCENE_KINDS:
            raise ControlError(f"{scene.id}: bilinmeyen tür '{kind}'. Geçerli: {', '.join(SCENE_KINDS)}")
        if preset not in presets:
            raise ControlError(f"{scene.id}: bilinmeyen preset '{preset}'. Geçerli: {', '.join(presets)}")
        if not text:
            raise ControlError(f"{scene.id}: metin boş.")
        scene.title, scene.kind, scene.preset, scene.text = title or scene.title, kind, preset, text
    return project


def build_app(service: PlaygroundService | None = None) -> gr.Blocks:
    service = service or PlaygroundService()
    narrator = SceneNarrator(service)
    default = VoxControls()

    with gr.Blocks(title="VoxForge Voice Playground") as app:
        gr.Markdown(
            "# VoxForge Voice Playground\n"
            "Türkçe metni seçtiğin motor ve kayıtlı ses profiliyle üret, dinle, indir ve karşılaştır. "
            "GPU'da aynı anda tek motor çalışır; motor değişince önceki kapatılır (ilk üretimde model yüklemesi sürer). "
            "8 GB VRAM'de VoxCPM2 için GPU kullanan diğer uygulamaları (canlı duvar kağıdı, tarayıcı, oyun istemcisi) kapatmak "
            "hızı belirgin artırır; her üretimin altında cihaz, süre, RTF, VRAM ve darboğaz notu görünür."
        )
        with gr.Tabs():
            # ================================ Playground ================================
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(label=f"Metin (en fazla {MAX_TEXT_CHARS} karakter)", value=SAMPLE_TEXT, lines=5, max_lines=14)
                        with gr.Row():
                            engine = gr.Radio(ENGINE_CHOICES, value="xtts", label="Motor")
                            chatterbox_setting = gr.Radio(CHATTERBOX_CHOICES, value="chatterbox-default", label="Chatterbox ayarı", visible=False)
                            voxcpm_setting = gr.Radio(VOXCPM_CHOICES, value="voxcpm2", label="VoxCPM2 ayarı", visible=False)
                        device = gr.Radio(
                            DEVICE_CHOICES,
                            value="cuda",
                            label="Cihaz (GPU varsayılan; CUDA yoksa üretim durur, CPU'ya kendiliğinden geçmez)",
                        )
                        reference_text = gr.Textbox(
                            label="Referansın birebir transkripti (yalnız VoxCPM2 Ultimate Cloning; profil başına hatırlanır)",
                            lines=3,
                            max_lines=8,
                            visible=False,
                            placeholder="Referans kayıtta ne söyleniyorsa aynen yazın; yanlış transkript benzerliği düşürür.",
                        )
                        with gr.Accordion("VoxCPM2 gelişmiş kontroller", open=False, visible=False) as advanced:
                            gr.Markdown(
                                "Yalnız **VoxCPM2 varsayılan** ve **Ultimate Cloning** için geçerlidir; *Hızlı (6 adım)* sabit "
                                "karşılaştırma varyantıdır ve bu paneli kullanmaz. Stil yönergesi seslendirilen metinden ayrı "
                                "tutulur ve modele `(yönerge)metin` biçiminde verilir (model kartındaki *controllable cloning*). "
                                "Model sayısal hız/duygu parametresi sunmaz; sonuçlar üretimden üretime değişebilir.  \n"
                                + ULTIMATE_NOTE
                            )
                            with gr.Row():
                                preset = gr.Dropdown(preset_choices(service), value="normal", label="Preset")
                                preset_delete = gr.Button("Kullanıcı preset'ini sil", size="sm")
                            preset_text = gr.Markdown(preset_info(service, "normal"))
                            with gr.Row():
                                cfg = gr.Slider(CFG_RANGE[0], CFG_RANGE[1], value=default.cfg, step=0.1, label="CFG (varsayılan 2.0)")
                                steps = gr.Slider(STEPS_RANGE[0], STEPS_RANGE[1], value=default.steps, step=1, label="Difüzyon adımı (varsayılan 10; <8 telaffuzu bozabilir)")
                            style = gr.Textbox(label="Stil yönergesi (boş = yok; ör. 'yavaş, sakin bir anlatım')", value="", max_lines=2)
                            pacing = gr.Radio(PACING_CHOICES, value=default.pacing, label="Anlatım ritmi")
                            with gr.Row():
                                sentence_pause = gr.Slider(PAUSE_RANGE[0], PAUSE_RANGE[1], value=default.sentence_pause, step=0.05, label="Cümle arası duraklama (sn, yalnız anlatım ritmi)")
                                paragraph_pause = gr.Slider(PAUSE_RANGE[0], PAUSE_RANGE[1], value=default.paragraph_pause, step=0.05, label="Paragraf arası duraklama (sn)")
                            with gr.Row():
                                preset_name = gr.Textbox(label="Yeni preset adı", max_lines=1, scale=3)
                                preset_save = gr.Button("Bu ayarları preset olarak kaydet", size="sm", scale=1)
                        with gr.Row():
                            profile = gr.Dropdown(profile_choices(), label="Ses profili", value=None, allow_custom_value=False)
                            refresh_profiles = gr.Button("Profilleri yenile", size="sm")
                        with gr.Row():
                            seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Tohum / seed (varsayılan 1234; -1 = rastgele)")
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
                        label="Varyantlar (motor sırasına göre üretilir, her model bir kez yüklenir; VoxCPM2 varsayılan/Ultimate panel ayarlarını kullanır)",
                    )
                    generate_many = gr.Button("Seçilenlerin hepsini üret")
                with gr.Accordion("Preset karşılaştırması (VoxCPM2): aynı metin, aynı ses, aynı tohum", open=False):
                    compare_presets = gr.CheckboxGroup(preset_choices(service), value=["normal", "yavas-anlatici"], label="Preset'ler")
                    generate_presets = gr.Button("Seçilen preset'leri üret")

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

            # ============================== Sahne anlatımı ==============================
            with gr.Tab("Sahne anlatımı"):
                gr.Markdown(
                    "Metni sahnelere ayır (`## Başlık`, `---` ya da `[sahne: Başlık]` işaretleri; işaret yoksa boş satırla "
                    "ayrılmış paragraflar). Her sahneye tür ve preset ata; tür yalnız bir kısayoldur ve varsayılan preset'i "
                    "seçer. Türler yerel anahtar kelimelerle *önerilir*, tablodan düzenlenebilir. Tüm sahneler **aynı profil, "
                    "aynı VoxCPM2 varyantı ve aynı tohumla** üretilir; yalnız stil, ritim ve duraklama değişir. Birleştirmede "
                    "kenar sessizlikleri kırpılır, sahneler arasına sabit duraklama konur ve ses seviyeleri ±6 dB içinde eşlenir.  \n"
                    "Türler/preset'ler: " + ", ".join(f"`{kind}` → `{preset}`" for kind, (_label, preset) in SCENE_KINDS.items())
                )
                with gr.Row():
                    scene_project_name = gr.Textbox(label="Proje adı", value="ilk-anlati", max_lines=1)
                    scene_existing = gr.Dropdown(narrator.store.list(), label="Kayıtlı sahne projeleri", value=None)
                    scene_load = gr.Button("Yükle", size="sm")
                with gr.Row():
                    scene_profile = gr.Dropdown(profile_choices(), label="Ses profili (tüm sahneler)", value=None)
                    scene_variant = gr.Radio(
                        [("VoxCPM2 (stil destekli)", "voxcpm2"), ("VoxCPM2 Ultimate (stil yok)", "voxcpm2-ultimate")],
                        value="voxcpm2", label="Varyant",
                    )
                    scene_seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Tohum (tüm sahneler)")
                    scene_gap = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Sahneler arası duraklama (sn)")
                scene_text = gr.Textbox(label="Anlatım metni", value=SAMPLE_SCENES, lines=10, max_lines=30)
                with gr.Row():
                    scene_suggest = gr.Checkbox(value=True, label="Tür/preset öner (yerel, LLM yok)")
                    scene_split = gr.Button("Sahnelere ayır", variant="secondary")
                scene_table = gr.Dataframe(
                    headers=SCENE_HEADERS, datatype=["number", "str", "str", "str", "str", "str"], interactive=True,
                    wrap=True, label="Sahneler (Başlık, Tür, Preset ve Metin düzenlenebilir)", col_count=(len(SCENE_HEADERS), "fixed"),
                )
                gr.Markdown(
                    "Geçerli türler: " + ", ".join(f"`{k}`" for k in SCENE_KINDS)
                    + " · preset anahtarları için Playground'daki preset listesine bakın (kullanıcı preset'leri `user-...`)."
                )
                with gr.Row():
                    scene_index = gr.Number(value=1, precision=0, label="Sahne no")
                    scene_generate_one = gr.Button("Bu sahneyi üret / yeniden üret")
                    scene_generate_all = gr.Button("Değişen/eksik sahneleri üret", variant="primary")
                    scene_merge = gr.Button("Tüm sahneleri tek WAV'da birleştir")
                with gr.Row():
                    scene_audio = gr.Audio(label="Seçili sahne", type="filepath", interactive=False)
                    scene_merged = gr.Audio(label="Birleşik anlatım", type="filepath", interactive=False)
                    scene_file = gr.File(label="Birleşik WAV indir", interactive=False)
                scene_message = gr.Markdown()
                scene_state = gr.State(None)

        # -- Playground handlers -----------------------------------------------------------
        def on_engine(value: str, voxcpm_value: str) -> list[Any]:
            is_vox = value == "voxcpm2"
            return [
                gr.update(visible=value == "chatterbox-v3"),
                gr.update(visible=is_vox),
                gr.update(visible=is_vox and voxcpm_value == "voxcpm2-ultimate"),
                gr.update(visible=is_vox),
            ]

        def on_voxcpm_setting(value: str) -> Any:
            return gr.update(visible=value == "voxcpm2-ultimate")

        def on_profile(slug: str | None) -> Any:
            return gr.update(value=service.get_transcript(slug) if slug else "")

        def on_preset(key: str | None) -> list[Any]:
            if not key:
                return [gr.update()] * 6 + [""]
            try:
                c = service.presets.get(key).controls
            except ControlError as exc:
                return [gr.update()] * 6 + [str(exc)]
            return [c.cfg, c.steps, c.style, c.pacing, c.sentence_pause, c.paragraph_pause, preset_info(service, key)]

        def on_preset_save(name, cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v):
            try:
                saved = service.presets.save(name, controls_from(cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v))
            except ControlError as exc:
                return gr.update(), gr.update(), f"Preset kaydedilmedi: {exc}"
            choices = preset_choices(service)
            return gr.update(choices=choices, value=saved.key), gr.update(choices=choices), f"Preset kaydedildi: **{saved.name}** (`{saved.key}`)."

        def on_preset_delete(key):
            try:
                service.presets.delete(key)
            except ControlError as exc:
                return gr.update(), gr.update(), str(exc)
            choices = preset_choices(service)
            return gr.update(choices=choices, value="normal"), gr.update(choices=choices), f"Silindi: `{key}`."

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

        def panel_controls(cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v) -> VoxControls:
            return controls_from(cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v)

        def on_generate(
            text_value, engine_value, chatter_value, voxcpm_value, device_value, transcript, profile_value, seed_value,
            normalize_value, preset_value, cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v, current,
        ):
            key = variant_for(engine_value, chatter_value, voxcpm_value)
            try:
                controls = panel_controls(cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v) if key in CONTROLLABLE_VARIANTS else None
            except ControlError as exc:
                return after_records([], current, [str(exc)])
            records, errors = service.generate_many(
                text_value, [key], profile_value, seed=_seed(seed_value), normalize=normalize_value,
                device=device_value or "cuda", reference_text=transcript, controls=controls,
                preset_key=preset_value if controls is not None and not controls.is_default() else None,
            )
            return after_records(records, current, errors)

        def on_generate_many(
            text_value, keys, device_value, transcript, profile_value, seed_value, normalize_value,
            cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v, current,
        ):
            if not keys:
                return after_records([], current, ["En az bir varyant seçin."])
            try:
                controls = panel_controls(cfg_v, steps_v, style_v, pacing_v, sp_v, pp_v)
            except ControlError as exc:
                return after_records([], current, [str(exc)])
            records, errors = service.generate_many(
                text_value, list(keys), profile_value, seed=_seed(seed_value), normalize=normalize_value,
                device=device_value or "cuda", reference_text=transcript, controls=controls,
            )
            return after_records(records, current, errors)

        def on_generate_presets(text_value, keys, device_value, profile_value, seed_value, normalize_value, current):
            if not keys:
                return after_records([], current, ["En az bir preset seçin."])
            records, errors = service.generate_presets(
                text_value, list(keys), profile_value, seed=_seed(seed_value), normalize=normalize_value, device=device_value or "cuda",
            )
            return after_records(records, current, errors)

        def on_unload():
            engine_name = service.manager.unload()
            note = f"{ENGINE_LABELS.get(engine_name, engine_name)} kapatıldı." if engine_name else "Yüklü motor yoktu."
            return status_line(service), note

        outputs = [result_audio, result_file, result_info, status, selected, message, *slot_components]
        panel = [cfg, steps, style, pacing, sentence_pause, paragraph_pause]
        engine.change(on_engine, [engine, voxcpm_setting], [chatterbox_setting, voxcpm_setting, reference_text, advanced])
        voxcpm_setting.change(on_voxcpm_setting, voxcpm_setting, reference_text)
        profile.change(on_profile, profile, reference_text)
        refresh_profiles.click(lambda: [gr.update(choices=profile_choices())] * 2, None, [profile, scene_profile])
        preset.change(on_preset, preset, [*panel, preset_text])
        preset_save.click(on_preset_save, [preset_name, *panel], [preset, compare_presets, preset_text])
        preset_delete.click(on_preset_delete, preset, [preset, compare_presets, preset_text])
        generate.click(
            on_generate,
            [text, engine, chatterbox_setting, voxcpm_setting, device, reference_text, profile, seed, normalize, preset, *panel, selected],
            outputs,
        )
        generate_many.click(
            on_generate_many, [text, compare_variants, device, reference_text, profile, seed, normalize, *panel, selected], outputs
        )
        generate_presets.click(
            on_generate_presets, [text, compare_presets, device, profile, seed, normalize, selected], outputs
        )
        selected.change(lambda ids: slot_updates(service, list(ids or [])), selected, slot_components)
        unload.click(on_unload, None, [status, message])
        app.load(lambda ids: slot_updates(service, list(ids or [])), selected, slot_components)

        # -- Scene handlers ----------------------------------------------------------------
        def scene_outputs(project: SceneProject | None, note: str, audio_path: Any = gr.update()) -> list[Any]:
            merged = project.merged_wav if project and project.merged_wav and Path(project.merged_wav).is_file() else None
            return [
                project,
                scene_rows(project, narrator),
                audio_path,
                merged,
                merged,
                note,
                gr.update(choices=narrator.store.list()),
            ]

        scene_out = [scene_state, scene_table, scene_audio, scene_merged, scene_file, scene_message, scene_existing]

        def sync_project(project, name, profile_value, variant_value, seed_value, gap_value, rows):
            if project is None:
                raise ControlError("Önce 'Sahnelere ayır' ya da kayıtlı bir projeyi yükleyin.")
            if (name or "").strip() != project.name and slugify(name) != project.slug:
                raise ControlError("Proje adı değişti; yeni proje için 'Sahnelere ayır'a basın.")
            apply_rows(project, rows, service.presets.all())
            project.profile = profile_value
            project.variant = variant_value or "voxcpm2"
            project.seed = _scene_seed(seed_value)
            project.scene_gap = float(gap_value)
            return project

        def on_scene_split(name, text_value, suggest, profile_value, variant_value, seed_value, gap_value):
            try:
                slug = slugify(name)
                scenes = split_scenes(text_value, suggest=bool(suggest))
                if not scenes:
                    raise ControlError("Metin boş.")
                project = SceneProject(
                    name=name.strip(), slug=slug, profile=profile_value, variant=variant_value or "voxcpm2",
                    seed=_scene_seed(seed_value), scene_gap=float(gap_value), scenes=scenes,
                )
                narrator.store.create(project)  # mevcut projenin üzerine asla yazmaz
            except ControlError as exc:
                return [gr.update()] + scene_outputs(None, f"Hata — {exc}")
            if project.slug != slug:
                where = f" `{slug}` adı kullanıldığı için mevcut proje korundu; yeni proje `{project.slug}` adıyla kaydedildi."
            else:
                where = f" `{project.slug}` olarak kaydedildi."
            return [project.name] + scene_outputs(
                project, f"{len(scenes)} sahne oluşturuldu.{where} Türleri/preset'leri düzenleyebilirsiniz."
            )

        def on_scene_load(slug):
            if not slug:
                return [gr.update()] * 6 + scene_outputs(None, "Bir proje seçin.")
            try:
                project = narrator.store.load(slug)
            except ControlError as exc:
                return [gr.update()] * 6 + scene_outputs(None, f"Hata — {exc}")
            loaded = [project.name, project.profile, project.variant, project.seed, project.scene_gap,
                      "\n\n".join(f"## {s.title}\n{s.text}" for s in project.scenes)]
            return loaded + scene_outputs(project, f"`{slug}` yüklendi.")

        def on_scene_generate_one(project, name, profile_value, variant_value, seed_value, gap_value, rows, index, device_value):
            try:
                project = sync_project(project, name, profile_value, variant_value, seed_value, gap_value, rows)
                position = int(index or 0)
                if not 1 <= position <= len(project.scenes):
                    raise ControlError(f"Sahne no 1-{len(project.scenes)} olmalı.")
                scene = narrator.generate_scene(project, project.scenes[position - 1].id, device=device_value or "cuda")
            except Exception as exc:  # noqa: BLE001 - kullanıcıya gösterilir
                narrator.store.save(project) if isinstance(project, SceneProject) else None
                return scene_outputs(project if isinstance(project, SceneProject) else None, f"Hata — {exc}")
            return scene_outputs(project, f"**{scene.title}** üretildi ({scene.audio_seconds} sn, preset `{scene.preset}`).", scene.wav)

        def on_scene_generate_all(project, name, profile_value, variant_value, seed_value, gap_value, rows, device_value):
            try:
                project = sync_project(project, name, profile_value, variant_value, seed_value, gap_value, rows)
            except ControlError as exc:
                return scene_outputs(project, f"Hata — {exc}")
            errors = narrator.generate_all(project, only_stale=True, device=device_value or "cuda")
            narrator.store.save(project)
            note = "Tüm sahneler hazır." if not errors else "\n\n".join(f"Hata — {e}" for e in errors)
            return scene_outputs(project, note)

        def on_scene_merge(project, name, profile_value, variant_value, seed_value, gap_value, rows, device_value):
            try:
                project = sync_project(project, name, profile_value, variant_value, seed_value, gap_value, rows)
                info = narrator.merge(project, device=device_value or "cuda")
            except ControlError as exc:
                return scene_outputs(project, f"Hata — {exc}")
            gains = ", ".join(f"{g:+.1f}" for g in info["gains_db"])
            return scene_outputs(project, f"Birleştirildi: {info['duration']:.1f} sn · seviye düzeltmesi (dB): {gains}")

        def on_scene_select(project, evt: gr.SelectData):
            if project is None or evt.index is None:
                return gr.update(), gr.update()
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            scene = project.scenes[row] if 0 <= row < len(project.scenes) else None
            return (scene.wav if scene and scene.wav and Path(scene.wav).is_file() else None), row + 1

        scene_common = [scene_state, scene_project_name, scene_profile, scene_variant, scene_seed, scene_gap, scene_table]
        scene_split.click(
            on_scene_split,
            [scene_project_name, scene_text, scene_suggest, scene_profile, scene_variant, scene_seed, scene_gap],
            [scene_project_name, *scene_out],
        )
        scene_load.click(
            on_scene_load, scene_existing,
            [scene_project_name, scene_profile, scene_variant, scene_seed, scene_gap, scene_text, *scene_out],
        )
        scene_generate_one.click(on_scene_generate_one, [*scene_common, scene_index, device], scene_out)
        scene_generate_all.click(on_scene_generate_all, [*scene_common, device], scene_out)
        scene_merge.click(on_scene_merge, [*scene_common, device], scene_out)
        scene_table.select(on_scene_select, scene_state, [scene_audio, scene_index])

    app.queue(default_concurrency_limit=1)
    return app


def _seed(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scene_seed(value: Any) -> int:
    """Scenes need one fixed seed (same voice in every scene); negative/empty -> default."""
    seed = _seed(value)
    return seed if seed is not None and seed >= 0 else DEFAULT_SEED


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
