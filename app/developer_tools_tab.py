# -*- coding: utf-8 -*-
"""Geliştirici araçları: XTTS fine-tuning hazırlığı (eski XTTS demosundan taşındı).

İşleyiciler ``app.gradio_xtts_demo`` içindeki mevcut fonksiyonlardır; burada yalnız ana
arayüzdeki yerleşim var. Hiçbiri model eğitmez: readiness raporu, experiment export ve
eğitim dry-run'ı alt süreç olarak mevcut betikleri çalıştırır. Eğitim başarısı yalnız gerçek
checkpoint ile kabul edilir (dry-run başarı sayılmaz).
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from app.gradio_xtts_demo import (
    NO_DATASET_VALUE,
    build_dataset_dropdown_choices,
    build_dataset_scan_status,
    export_experiment_from_gradio,
    run_readiness_report_from_gradio,
    run_training_dry_run_from_gradio,
    scan_finetune_datasets,
)
from app.ui_theme import lead, section


def refresh_datasets(current: str | None) -> list[Any]:
    scan = scan_finetune_datasets()
    choices = build_dataset_dropdown_choices(scan)
    values = {value for _label, value in choices}
    return [gr.update(choices=choices, value=current if current in values else NO_DATASET_VALUE), build_dataset_scan_status(scan)]


def build_developer_tools() -> dict[str, Any]:
    """Render the section inside the current Blocks context; returns its components."""
    scan = scan_finetune_datasets()
    lead(
        "XTTS fine-tuning hazırlığı. Bu bölüm model eğitmez: dataset hazır mı, experiment klasörü "
        "doğru mu ve eğitim komutu kurulabiliyor mu kontrol eder. `datasets/` ve `experiments/` yerel kalır."
    )
    with gr.Group(elem_classes="vf-card"):
        section("Dataset ve deney")
        with gr.Row(equal_height=True):
            dataset = gr.Dropdown(
                label="Fine-tuning dataset", choices=build_dataset_dropdown_choices(scan),
                value=NO_DATASET_VALUE, interactive=True, scale=4,
            )
            refresh = gr.Button("Datasetleri yenile", size="sm", scale=1, min_width=140)
        dataset_status = gr.Markdown(build_dataset_scan_status(scan), elem_classes="vf-status")
        run_name = gr.Textbox(label="Deney adı (run name)", placeholder="Örnek: baglare_xtts_exp01", max_lines=1)
    with gr.Accordion("Dry-run eğitim ayarları", open=False):
        with gr.Row():
            max_steps = gr.Number(label="Max steps", value=300, precision=0)
            epochs = gr.Number(label="Epochs", value=1, precision=0)
            batch_size = gr.Number(label="Batch size", value=1, precision=0)
        with gr.Row():
            grad_accum = gr.Number(label="Grad accumulation", value=16, precision=0)
            save_step = gr.Number(label="Save step", value=1, precision=0)
    with gr.Row():
        readiness = gr.Button("Readiness raporu çalıştır")
        export = gr.Button("Experiment export et")
        dry_run = gr.Button("Training dry-run çalıştır", variant="primary")
    readiness_status = gr.Markdown("Readiness sonucu burada görünecek.")
    export_status = gr.Markdown("Export sonucu burada görünecek.")
    dry_run_status = gr.Markdown("Dry-run sonucu burada görünecek. Bu işlem eğitim başlatmaz.")

    refresh.click(refresh_datasets, dataset, [dataset, dataset_status])
    readiness.click(run_readiness_report_from_gradio, dataset, readiness_status)
    export.click(export_experiment_from_gradio, [dataset, run_name], export_status)
    dry_run.click(
        run_training_dry_run_from_gradio,
        [dataset, run_name, max_steps, epochs, batch_size, grad_accum, save_step],
        dry_run_status,
    )
    return {"dataset": dataset, "dataset_status": dataset_status, "refresh": refresh}
