# -*- coding: utf-8 -*-
"""Shared VoxForge look: Gradio theme, CSS and small layout helpers.

A paper-toned writing surface with olive controls and a quieter dark variant.
Only presentation lives here; no handler or data logic.
"""

from __future__ import annotations

import html

import gradio as gr

OLIVE = gr.themes.Color(
    c50="#f4f6ef", c100="#e9eddf", c200="#d5dfc7", c300="#b9cba7", c400="#9cb785",
    c500="#829e6d", c600="#668353", c700="#4e6942", c800="#405638", c900="#34462f", c950="#202e1f",
    name="vf_olive",
)
CHARCOAL = gr.themes.Color(
    c50="#f7f5f2", c100="#efebe5", c200="#ddd7ce", c300="#c2bab0", c400="#9c9389",
    c500="#68675d", c600="#5d5650", c700="#3b3632", c800="#262320", c900="#242621", c950="#1c1e1a",
    name="vf_charcoal",
)

FONT = (gr.themes.Font("Geist"), "ui-sans-serif", "system-ui", "sans-serif")
FONT_MONO = (gr.themes.Font("Geist Mono"), "ui-monospace", "Consolas", "monospace")


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=OLIVE,
        secondary_hue=OLIVE,
        neutral_hue=CHARCOAL,
        font=FONT,
        font_mono=FONT_MONO,
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_lg,
    )
    return theme.set(
        # page and surfaces
        body_background_fill="#f6f5f0",
        body_background_fill_dark="#1c1e1a",
        body_text_color="#1f1c19",
        body_text_color_dark="#ece6de",
        body_text_color_subdued="#6a6159",
        body_text_color_subdued_dark="#a69d93",
        background_fill_primary="#fffefa",
        background_fill_primary_dark="#242621",
        background_fill_secondary="#f0ece6",
        background_fill_secondary_dark="#171513",
        block_background_fill="#fffefa",
        block_background_fill_dark="#242621",
        block_border_color="#deded4",
        block_border_color_dark="#404239",
        block_border_width="1px",
        block_shadow="none",
        block_shadow_dark="none",
        block_radius="12px",
        block_padding="14px 16px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color="#6a6159",
        block_label_text_color_dark="#a69d93",
        block_label_text_size="16px",
        block_label_text_weight="500",
        block_title_text_color="#3a342f",
        block_title_text_color_dark="#d8d0c6",
        block_title_text_size="16px",
        block_title_text_weight="500",
        block_info_text_color="#68675d",
        block_info_text_color_dark="#958b81",
        block_info_text_size="15px",
        panel_background_fill="#f0ece6",
        panel_background_fill_dark="#171513",
        panel_border_color="#deded4",
        panel_border_color_dark="#404239",
        border_color_primary="#deded4",
        border_color_primary_dark="#404239",
        border_color_accent="#4e6942",
        border_color_accent_dark="#b9cba7",
        color_accent="#4e6942",
        color_accent_soft="#e9eddf",
        color_accent_soft_dark="#303c2e",
        layout_gap="14px",
        # inputs
        input_background_fill="#ffffff",
        input_background_fill_dark="#20221e",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#20221e",
        input_border_color="#d9d2c8",
        input_border_color_dark="#34302c",
        input_border_color_focus="#4e6942",
        input_border_color_focus_dark="#b9cba7",
        input_border_width="1px",
        input_radius="8px",
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 3px rgba(78, 105, 66, 0.16)",
        input_shadow_focus_dark="0 0 0 3px rgba(185, 203, 167, 0.18)",
        input_placeholder_color="#706e64",
        input_placeholder_color_dark="#aaa99e",
        # checkbox / radio / slider
        checkbox_background_color="#ffffff",
        checkbox_background_color_dark="#20221e",
        checkbox_background_color_selected="#4e6942",
        checkbox_background_color_selected_dark="#b9cba7",
        checkbox_border_color="#c9c1b6",
        checkbox_border_color_dark="#4a443e",
        checkbox_border_color_selected="#4e6942",
        checkbox_border_color_selected_dark="#b9cba7",
        checkbox_label_background_fill="#f7f4ef",
        checkbox_label_background_fill_dark="#1f1c1a",
        checkbox_label_background_fill_hover="#f0ece6",
        checkbox_label_background_fill_hover_dark="#25221f",
        checkbox_label_background_fill_selected="#e9eddf",
        checkbox_label_background_fill_selected_dark="#303c2e",
        checkbox_label_border_color="#deded4",
        checkbox_label_border_color_dark="#2f2b27",
        checkbox_label_border_color_selected="#4e6942",
        checkbox_label_border_color_selected_dark="#829e6d",
        checkbox_label_text_color_selected="#34462f",
        checkbox_label_text_color_selected_dark="#d5dfc7",
        checkbox_label_shadow="none",
        slider_color="#4e6942",
        slider_color_dark="#b9cba7",
        # buttons
        button_large_radius="10px",
        button_small_radius="8px",
        button_large_text_weight="600",
        button_small_text_weight="500",
        button_small_text_size="16px",
        button_large_text_size="18px",
        button_primary_background_fill="#4e6942",
        button_primary_background_fill_hover="#405638",
        button_primary_background_fill_dark="#b9cba7",
        button_primary_background_fill_hover_dark="#d5dfc7",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#202e1f",
        button_primary_border_color="#4e6942",
        button_primary_border_color_dark="#b9cba7",
        button_primary_shadow="0 1px 2px rgba(60, 30, 5, 0.18)",
        button_primary_shadow_dark="none",
        button_primary_shadow_hover_dark="none",
        button_secondary_shadow="none",
        button_secondary_shadow_dark="none",
        button_secondary_shadow_hover="none",
        button_secondary_shadow_hover_dark="none",
        button_secondary_shadow_active="none",
        button_secondary_shadow_active_dark="none",
        button_secondary_background_fill="#efeae3",
        button_secondary_background_fill_hover="#e6dfd6",
        button_secondary_background_fill_dark="#26231f",
        button_secondary_background_fill_hover_dark="#302c28",
        button_secondary_text_color="#2a2521",
        button_secondary_text_color_dark="#e4ddd4",
        button_secondary_border_color="#ddd5ca",
        button_secondary_border_color_dark="#36312c",
        button_cancel_background_fill="#f6e3dd",
        button_cancel_background_fill_dark="#3a1d17",
        button_cancel_text_color="#8c2b17",
        button_cancel_text_color_dark="#f5b4a4",
        # tables
        table_border_color="#deded4",
        table_border_color_dark="#404239",
        table_even_background_fill="#fffefa",
        table_even_background_fill_dark="#242621",
        table_odd_background_fill="#f6f3ee",
        table_odd_background_fill_dark="#191715",
        table_row_focus="#e9eddf",
        table_row_focus_dark="#303c2e",
        link_text_color="#4e6942",
        link_text_color_dark="#d5dfc7",
        link_text_color_hover="#405638",
        link_text_color_hover_dark="#e9eddf",
        code_background_fill="#efeae3",
        code_background_fill_dark="#24211e",
    )


THEME = build_theme()

CSS = """
:root {
  --vf-accent: #4e6942;
  --vf-accent-soft: #e9eddf;
  --vf-ink: #1f1c19;
  --vf-muted: #6a6159;
  --vf-line: #deded4;
  --vf-panel: #fffefa;
  --vf-sunk: #f0ece6;
  --vf-ok: #2f7a4a;
  --vf-warn: #9a5b00;
}
.dark {
  --vf-accent: #b9cba7;
  --vf-accent-soft: #303c2e;
  --vf-ink: #ece6de;
  --vf-muted: #a69d93;
  --vf-line: #404239;
  --vf-panel: #242621;
  --vf-sunk: #171513;
  --vf-ok: #7fcf98;
  --vf-warn: #f3c06b;
}

.gradio-container { box-sizing: border-box; width: 100% !important; min-width: 0; max-width: 1040px !important; margin: 0 auto !important; padding: 0 20px 48px !important; }
.gradio-container ::selection { background: var(--vf-accent-soft); color: var(--vf-ink); }
.gradio-container { caret-color: var(--vf-accent); accent-color: var(--vf-accent); }
.gradio-container * { scrollbar-width: thin; scrollbar-color: var(--vf-line) transparent; }
.gradio-container :focus-visible { outline: 2px solid var(--vf-accent); outline-offset: 2px; }
.gradio-container main.fillable { width: 100%; min-width: 0; padding: 0 !important; }
footer { opacity: 0.55; }

/* --- masthead ---------------------------------------------------------- */
.vf-hero { display: flex; align-items: center; gap: 12px; padding: 28px 0 18px; }
.vf-mark { color: var(--vf-accent); display: grid; place-items: center; width: 38px; height: 38px;
  border-radius: 50%; background: var(--vf-accent-soft); }
.vf-mark svg { width: 22px; }
.vf-mark rect { fill: var(--vf-accent); }
.vf-hero h1 { margin: 0; font: 600 26px/1.2 var(--font); letter-spacing: -.035em; color: var(--vf-ink); }
.vf-hero h1 span { margin-left: 12px; padding-left: 14px; border-left: 1px solid var(--vf-line);
  font: 400 16px/1.5 var(--font); letter-spacing: normal; color: var(--vf-muted); }
.vf-tabs .tab-container { gap: 3px; border-bottom: 1px solid var(--vf-line); }
.vf-tabs button[role="tab"] { padding: 14px 20px; font-size: 19px; color: var(--vf-muted); }
.vf-tabs button[role="tab"][aria-selected="true"] { color: var(--vf-accent); font-weight: 600; }
.vf-tabs [role="tabpanel"] { border: 0; padding: clamp(16px, 3vw, 36px) 0; }
.vf-sheet { padding: clamp(18px, 3vw, 36px) !important; background: var(--vf-panel);
  border: 1px solid var(--vf-line); border-radius: 20px; gap: 18px !important; }
.vf-sheet .form, .vf-sheet .block { border: 0 !important; background: transparent !important; box-shadow: none !important; }
.vf-sheet .block { padding: 0 !important; }
.vf-sheet .vf-section { margin: 0; }
.vf-sheet .form { gap: 12px; }
.vf-sheet .vf-manuscript textarea { background: var(--vf-panel); }
.vf-sheet .row { align-items: flex-end; }
.vf-sheet .vf-section h3 { font: 400 clamp(34px, 4vw, 48px)/1.12 Georgia, serif; letter-spacing: -.035em; }
.vf-manuscript textarea { font: 400 21px/1.65 Georgia, serif; padding: 14px 16px; }
.vf-sheet .vf-go { min-height: 50px !important; margin-bottom: 0 !important; }
.vf-tabs [role="tabpanel"]:first-of-type { min-width: 0; }
/* Main destinations, project types, then the steps within a book. */
.vf-tabs > .tab-wrapper button[role="tab"] { font-weight: 600; min-height: 50px; }
.vf-modes > .tab-wrapper, .vf-steps > .tab-wrapper { width: 100%; min-width: 0; }
.vf-modes > .tab-wrapper .tab-container { border-bottom: 0; gap: 4px; }
.vf-modes > .tab-wrapper button[role="tab"] { border: 1px solid var(--vf-line); border-radius: 8px; }
.vf-modes > .tab-wrapper button[aria-selected="true"] { background: var(--vf-accent-soft); }
.vf-steps > .tab-wrapper button[role="tab"] { padding: 10px 14px; }
.vf-tabs .vf-modes [role="tabpanel"], .vf-tabs .vf-steps [role="tabpanel"] { padding: 12px 0; }
/* --- section headings and lead text ------------------------------------------ */
.vf-section { margin: 10px 0 -4px; }
.vf-section h3 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; color: var(--vf-ink); }
.vf-section p { margin: 2px 0 0; font-size: 15px; color: var(--vf-muted); max-width: 72ch; }
.vf-lead { color: var(--vf-muted); font-size: 17px; max-width: 75ch; margin: 4px 0 2px; }
.vf-lead p { margin: 0; }
.vf-hint, .vf-hint p { color: var(--vf-muted) !important; font-size: 15px !important; line-height: 1.5; }

/* --- cards: a Group/Column that holds one step of the task --------------------- */
.vf-card { border: 1px solid var(--vf-line) !important; border-radius: 14px !important; background: var(--vf-panel) !important; padding: 6px !important; gap: 0 !important; }
.vf-card > .styler { background: transparent !important; gap: 0 !important; }
.vf-card .block { border: 0 !important; background: transparent !important; box-shadow: none !important; }
.vf-card .html-container.padding { padding: 0 !important; }
.vf-card .vf-section { padding: 12px 16px 0; margin: 0; }
.vf-card .form { border: 0 !important; background: transparent !important; }
.vf-card .styler > button { margin: 2px 10px 12px; }
.vf-card span.md { display: block; padding: 4px 16px; }
.vf-card .row > button { margin-left: 10px; margin-right: 10px; }

/* a button that sits beside a labelled input lines up with the input, not the label */
.gradio-container .row.stretch > button { align-self: flex-end; margin-bottom: 14px; height: auto; min-height: 46px; }

/* result rail stays in view while the composer scrolls */
.vf-rail { position: sticky; top: 12px; align-self: flex-start; }
.vf-rail .vf-card { background: var(--vf-sunk) !important; }

/* engine/GPU status line: quiet, monospace numbers */
.vf-status, .vf-status p { font-size: 15px !important; color: var(--vf-muted) !important; font-variant-numeric: tabular-nums; }
.vf-status strong { color: var(--vf-ink); }
.vf-progress { border: 1px solid var(--vf-line); border-radius: 12px; background: var(--vf-sunk); padding: 12px 16px !important; }
.vf-progress p { margin: 2px 0; }

/* per-generation technical details under compare slots: small and quiet */
.vf-slot, .vf-slot p, .vf-slot li { font-size: 15px !important; line-height: 1.45; color: var(--vf-muted) !important; }
.vf-slot strong { color: var(--vf-ink); }

/* the one action that matters on each surface */
.vf-go { min-height: 48px !important; font-size: 18px !important; letter-spacing: -0.005em; }

/* compact file box instead of a 200px empty drop area */
.vf-file { height: auto !important; min-height: 0 !important; }
.vf-file .empty { height: 64px !important; min-height: 0 !important; }
.vf-file .wrap { min-height: 0 !important; }

/* empty player in the result rail reads as a waiting slot, not a missing component */
.vf-rail .empty.small { min-height: 88px; margin: 10px 16px; border: 1px dashed var(--vf-line); border-radius: 10px; opacity: 0.8; }

/* accordions read as quiet disclosure rows */
.gradio-container .label-wrap { font-weight: 550; }
.gradio-container .label-wrap span { font-size: 17px; }

/* data tables: tabular numerals */
.gradio-container table { font-variant-numeric: tabular-nums; }

/* phone */
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px 32px !important; }
  .vf-hero { flex-wrap: wrap; padding-top: 18px; }
  .vf-hero h1 { font-size: 25px; }
  .vf-sheet .vf-go { align-self: stretch; }
  .vf-rail { position: static; }
  .vf-tabs button[role="tab"] { padding: 10px 12px; font-size: 17px; }
}
"""

_MARK = (
    '<svg viewBox="0 0 28 28" aria-hidden="true">'
    '<rect x="2" y="10" width="3.2" height="8" rx="1.6"/>'
    '<rect x="7.6" y="5" width="3.2" height="18" rx="1.6"/>'
    '<rect x="13.2" y="2" width="3.2" height="24" rx="1.6"/>'
    '<rect x="18.8" y="7" width="3.2" height="14" rx="1.6"/>'
    '<rect x="24.4" y="11" width="3.2" height="6" rx="1.6"/>'
    "</svg>"
)


def hero(title: str, subtitle: str) -> gr.HTML:
    """Compact masthead shared by both local applications."""
    return gr.HTML(
        f'<header class="vf-hero"><div class="vf-mark">{_MARK}</div>'
        f"<h1>{html.escape(title)}<span>{html.escape(subtitle)}</span></h1></header>"
    )


def section(title: str, hint: str | None = None) -> gr.HTML:
    """Small heading for a step inside a card or page."""
    body = f"<h3>{html.escape(title)}</h3>" + (f"<p>{html.escape(hint)}</p>" if hint else "")
    return gr.HTML(f'<div class="vf-section">{body}</div>')


def lead(markdown: str) -> gr.Markdown:
    """One short paragraph under a tab heading."""
    return gr.Markdown(markdown, elem_classes="vf-lead")
