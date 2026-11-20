# -*- coding: utf-8 -*-
"""Shared VoxForge look: Gradio theme, CSS and small layout helpers.

One visual world for every VoxForge Gradio app (Voice Playground, Studio, XTTS demo):
a warm charcoal "mixing desk" surface, a single amber signal colour and Geist type.
Light mode is supported with the same roles; Gradio switches on the system setting.
Only presentation lives here; no handler or data logic.
"""

from __future__ import annotations

import html

import gradio as gr

# Amber signal colour (VU-meter / record lamp). c500 is the dark-mode accent,
# c700 the light-mode accent (contrast on paper).
AMBER = gr.themes.Color(
    c50="#fff8eb", c100="#feecc7", c200="#fdd68a", c300="#fcbc4d", c400="#f7a52a",
    c500="#f0901a", c600="#d26e0e", c700="#ab500f", c800="#8a3f13", c900="#723513", c950="#411a06",
    name="vf_amber",
)
# Warm charcoal neutrals (slightly brown, never blue-grey).
CHARCOAL = gr.themes.Color(
    c50="#f7f5f2", c100="#efebe5", c200="#ddd7ce", c300="#c2bab0", c400="#9c9389",
    c500="#7a7168", c600="#5d5650", c700="#3b3632", c800="#262320", c900="#1b1917", c950="#121110",
    name="vf_charcoal",
)

FONT = (gr.themes.GoogleFont("Geist"), "ui-sans-serif", "system-ui", "sans-serif")
FONT_MONO = (gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "Consolas", "monospace")


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=AMBER,
        secondary_hue=AMBER,
        neutral_hue=CHARCOAL,
        font=FONT,
        font_mono=FONT_MONO,
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    )
    return theme.set(
        # page and surfaces
        body_background_fill="#f4f1ec",
        body_background_fill_dark="#121110",
        body_text_color="#1f1c19",
        body_text_color_dark="#ece6de",
        body_text_color_subdued="#6a6159",
        body_text_color_subdued_dark="#a69d93",
        background_fill_primary="#fbfaf7",
        background_fill_primary_dark="#1b1917",
        background_fill_secondary="#f0ece6",
        background_fill_secondary_dark="#171513",
        block_background_fill="#fbfaf7",
        block_background_fill_dark="#1b1917",
        block_border_color="#e2dcd3",
        block_border_color_dark="#2c2825",
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
        block_label_text_size="13px",
        block_label_text_weight="500",
        block_title_text_color="#3a342f",
        block_title_text_color_dark="#d8d0c6",
        block_title_text_size="13px",
        block_title_text_weight="500",
        block_info_text_color="#7a7168",
        block_info_text_color_dark="#958b81",
        block_info_text_size="12.5px",
        panel_background_fill="#f0ece6",
        panel_background_fill_dark="#171513",
        panel_border_color="#e2dcd3",
        panel_border_color_dark="#2c2825",
        border_color_primary="#e2dcd3",
        border_color_primary_dark="#2c2825",
        border_color_accent="#ab500f",
        border_color_accent_dark="#f0901a",
        color_accent="#ab500f",
        color_accent_soft="#feecc7",
        color_accent_soft_dark="#2e2114",
        layout_gap="14px",
        # inputs
        input_background_fill="#ffffff",
        input_background_fill_dark="#141210",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#141210",
        input_border_color="#d9d2c8",
        input_border_color_dark="#34302c",
        input_border_color_focus="#ab500f",
        input_border_color_focus_dark="#f0901a",
        input_border_width="1px",
        input_radius="8px",
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 3px rgba(171, 80, 15, 0.16)",
        input_shadow_focus_dark="0 0 0 3px rgba(240, 144, 26, 0.18)",
        input_placeholder_color="#8a8178",
        input_placeholder_color_dark="#7d746b",
        # checkbox / radio / slider
        checkbox_background_color="#ffffff",
        checkbox_background_color_dark="#141210",
        checkbox_background_color_selected="#ab500f",
        checkbox_background_color_selected_dark="#f0901a",
        checkbox_border_color="#c9c1b6",
        checkbox_border_color_dark="#4a443e",
        checkbox_border_color_selected="#ab500f",
        checkbox_border_color_selected_dark="#f0901a",
        checkbox_label_background_fill="#f7f4ef",
        checkbox_label_background_fill_dark="#1f1c1a",
        checkbox_label_background_fill_hover="#f0ece6",
        checkbox_label_background_fill_hover_dark="#25221f",
        checkbox_label_background_fill_selected="#fdf1dc",
        checkbox_label_background_fill_selected_dark="#2e2114",
        checkbox_label_border_color="#e2dcd3",
        checkbox_label_border_color_dark="#2f2b27",
        checkbox_label_border_color_selected="#ab500f",
        checkbox_label_border_color_selected_dark="#b56a14",
        checkbox_label_text_color_selected="#5c2a08",
        checkbox_label_text_color_selected_dark="#fcd9a4",
        checkbox_label_shadow="none",
        slider_color="#ab500f",
        slider_color_dark="#f0901a",
        # buttons
        button_large_radius="10px",
        button_small_radius="8px",
        button_large_text_weight="600",
        button_small_text_weight="500",
        button_primary_background_fill="#ab500f",
        button_primary_background_fill_hover="#8a3f13",
        button_primary_background_fill_dark="#f0901a",
        button_primary_background_fill_hover_dark="#f7a52a",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#1b1006",
        button_primary_border_color="#ab500f",
        button_primary_border_color_dark="#f0901a",
        button_primary_shadow="0 1px 2px rgba(60, 30, 5, 0.18)",
        button_primary_shadow_dark="0 6px 18px -8px rgba(240, 144, 26, 0.55)",
        button_primary_shadow_hover_dark="0 8px 22px -8px rgba(240, 144, 26, 0.65)",
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
        table_border_color="#e2dcd3",
        table_border_color_dark="#2c2825",
        table_even_background_fill="#fbfaf7",
        table_even_background_fill_dark="#1b1917",
        table_odd_background_fill="#f6f3ee",
        table_odd_background_fill_dark="#191715",
        table_row_focus="#fdf1dc",
        table_row_focus_dark="#2e2114",
        link_text_color="#ab500f",
        link_text_color_dark="#f7a52a",
        link_text_color_hover="#8a3f13",
        link_text_color_hover_dark="#fcbc4d",
        code_background_fill="#efeae3",
        code_background_fill_dark="#24211e",
    )


THEME = build_theme()

CSS = """
:root {
  --vf-accent: #ab500f;
  --vf-accent-soft: #fdf1dc;
  --vf-ink: #1f1c19;
  --vf-muted: #6a6159;
  --vf-line: #e2dcd3;
  --vf-panel: #fbfaf7;
  --vf-sunk: #f0ece6;
  --vf-ok: #2f7a4a;
  --vf-warn: #9a5b00;
}
.dark {
  --vf-accent: #f0901a;
  --vf-accent-soft: #2e2114;
  --vf-ink: #ece6de;
  --vf-muted: #a69d93;
  --vf-line: #2c2825;
  --vf-panel: #1b1917;
  --vf-sunk: #171513;
  --vf-ok: #7fcf98;
  --vf-warn: #f3c06b;
}

.gradio-container { max-width: 1320px !important; margin: 0 auto !important; padding: 0 20px 48px !important; }
.gradio-container ::selection { background: var(--vf-accent); color: #1b1006; }
.gradio-container { caret-color: var(--vf-accent); accent-color: var(--vf-accent); }
.gradio-container * { scrollbar-width: thin; scrollbar-color: var(--vf-line) transparent; }
.gradio-container :focus-visible { outline: 2px solid var(--vf-accent); outline-offset: 2px; }
footer { opacity: 0.55; }

/* --- masthead ---------------------------------------------------------- */
.vf-hero { display: flex; align-items: center; gap: 18px; padding: 26px 2px 18px; border-bottom: 1px solid var(--vf-line); margin-bottom: 6px; }
.vf-hero .vf-mark { flex: none; width: 46px; height: 46px; border-radius: 12px; background: var(--vf-accent-soft); display: grid; place-items: center; }
.vf-hero .vf-mark svg { width: 28px; height: 28px; }
.vf-hero .vf-mark rect { fill: var(--vf-accent); transform-box: fill-box; transform-origin: center; animation: vf-level 1.1s cubic-bezier(0.16, 1, 0.3, 1) both; }
.vf-hero .vf-mark rect:nth-child(2) { animation-delay: 60ms; }
.vf-hero .vf-mark rect:nth-child(3) { animation-delay: 120ms; }
.vf-hero .vf-mark rect:nth-child(4) { animation-delay: 180ms; }
.vf-hero .vf-mark rect:nth-child(5) { animation-delay: 240ms; }
@keyframes vf-level { from { transform: scaleY(0.35); opacity: 0.6; } to { transform: scaleY(1); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .vf-hero .vf-mark rect { animation: none; } }
.vf-hero h1 { margin: 0; font-size: 26px; line-height: 1.15; font-weight: 650; letter-spacing: -0.025em; color: var(--vf-ink); }
.vf-hero h1 span { color: var(--vf-muted); font-weight: 450; }
.vf-hero p { margin: 4px 0 0; color: var(--vf-muted); font-size: 14.5px; max-width: 70ch; text-wrap: pretty; }
.vf-hero .vf-chips { margin-left: auto; display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.vf-chip { font: 500 12px/1 var(--font-mono); color: var(--vf-muted); border: 1px solid var(--vf-line); border-radius: 999px; padding: 6px 10px; white-space: nowrap; }

/* --- tabs ---------------------------------------------------------------- */
.vf-tabs > .tab-wrapper, .vf-tabs .tab-container { border-bottom: 1px solid var(--vf-line); }
.vf-tabs button[role="tab"] { font-size: 14.5px; font-weight: 500; padding: 12px 16px; color: var(--vf-muted); }
.vf-tabs button[role="tab"][aria-selected="true"] { color: var(--vf-ink); font-weight: 600; }
.vf-tabs button[role="tab"]:hover { color: var(--vf-ink); }

/* --- section headings and lead text ------------------------------------------ */
.vf-section { margin: 10px 0 -4px; }
.vf-section h3 { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: var(--vf-ink); }
.vf-section p { margin: 2px 0 0; font-size: 13px; color: var(--vf-muted); max-width: 72ch; }
.vf-lead { color: var(--vf-muted); font-size: 14px; max-width: 75ch; margin: 4px 0 2px; }
.vf-lead p { margin: 0; }
.vf-hint, .vf-hint p { color: var(--vf-muted) !important; font-size: 12.5px !important; line-height: 1.5; }

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
.gradio-container .row.stretch > button { align-self: flex-end; margin-bottom: 14px; height: auto; min-height: 40px; }

/* result rail stays in view while the composer scrolls */
.vf-rail { position: sticky; top: 12px; align-self: flex-start; }
.vf-rail .vf-card { background: var(--vf-sunk) !important; }

/* engine/GPU status line: quiet, monospace numbers */
.vf-status, .vf-status p { font-size: 12.5px !important; color: var(--vf-muted) !important; font-variant-numeric: tabular-nums; }
.vf-status strong { color: var(--vf-ink); }
.vf-progress { border: 1px solid var(--vf-line); border-radius: 12px; background: var(--vf-sunk); padding: 12px 16px !important; }
.vf-progress p { margin: 2px 0; }

/* per-generation technical details under compare slots: small and quiet */
.vf-slot, .vf-slot p, .vf-slot li { font-size: 12px !important; line-height: 1.45; color: var(--vf-muted) !important; }
.vf-slot strong { color: var(--vf-ink); }

/* the one action that matters on each surface */
.vf-go { min-height: 48px !important; font-size: 15.5px !important; letter-spacing: -0.005em; }

/* compact file box instead of a 200px empty drop area */
.vf-file { height: auto !important; min-height: 0 !important; }
.vf-file .empty { height: 64px !important; min-height: 0 !important; }
.vf-file .wrap { min-height: 0 !important; }

/* empty player in the result rail reads as a waiting slot, not a missing component */
.vf-rail .empty.small { min-height: 88px; margin: 10px 16px; border: 1px dashed var(--vf-line); border-radius: 10px; opacity: 0.8; }

/* accordions read as quiet disclosure rows */
.gradio-container .label-wrap { font-weight: 550; }
.gradio-container .label-wrap span { font-size: 14px; }

/* data tables: tabular numerals */
.gradio-container table { font-variant-numeric: tabular-nums; }

/* phone */
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px 32px !important; }
  .vf-hero { flex-wrap: wrap; padding-top: 18px; }
  .vf-hero h1 { font-size: 22px; }
  .vf-hero .vf-chips { margin-left: 0; justify-content: flex-start; }
  .vf-rail { position: static; }
  .vf-tabs button[role="tab"] { padding: 10px 12px; font-size: 14px; }
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


def hero(title: str, subtitle: str, tagline: str, chips: list[str] | None = None) -> gr.HTML:
    """Masthead: waveform mark, product name, one-line purpose and quiet fact chips."""
    chip_html = "".join(f'<span class="vf-chip">{html.escape(chip)}</span>' for chip in chips or [])
    return gr.HTML(
        f'<header class="vf-hero">'
        f'<div class="vf-mark">{_MARK}</div>'
        f"<div><h1>{html.escape(title)} <span>{html.escape(subtitle)}</span></h1>"
        f"<p>{html.escape(tagline)}</p></div>"
        f'<div class="vf-chips">{chip_html}</div>'
        f"</header>"
    )


def section(title: str, hint: str | None = None) -> gr.HTML:
    """Small heading for a step inside a card or page."""
    body = f"<h3>{html.escape(title)}</h3>" + (f"<p>{html.escape(hint)}</p>" if hint else "")
    return gr.HTML(f'<div class="vf-section">{body}</div>')


def lead(markdown: str) -> gr.Markdown:
    """One short paragraph under a tab heading."""
    return gr.Markdown(markdown, elem_classes="vf-lead")
