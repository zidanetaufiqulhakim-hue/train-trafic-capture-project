"""
Train Traffic Capture — Gradio application.

Run with:
  python application/app.py

Flow:
  1. App starts  — models warm up automatically (YOLO + EasyOCR)
  2. Add video   — upload a local file or paste a YouTube URL
  3. Run analysis — pipeline processes frames with a live progress indicator
  4. Explore     — raw data table, station summary, interactive charts
  5. Export      — download the CSV and the chart image
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Hosted deployments (e.g. HF Spaces) set DISABLE_PNG_EXPORT=1 because the
# Kaleido PNG export needs a Chrome binary we don't install there. Locally,
# the Download chart (PNG) button stays visible.
_PNG_EXPORT_ENABLED = os.environ.get("DISABLE_PNG_EXPORT", "0") != "1"

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr
import pandas as pd

import application.config as config
from application.pipeline.runner import Models, iter_pipeline, load_models
from application.pipeline.video_io import download_youtube_video, is_youtube_url
from application.analytics.aggregation import (
    aggregate_by_station,
    aggregate_by_time_interval,
    aggregate_heatmap,
    aggregate_time_series,
)
from application.analytics.charts import (
    figure_to_bytes,
    passenger_heatmap_chart,
    passenger_pct_stacked_bar,
    passenger_timeline_chart,
    passenger_volume_bar,
    station_share_pie,
)

# ── Global model state ────────────────────────────────────────────────────────

_models: Models | None = None


def _warmup() -> str:
    global _models
    model_path = config.YOLO_MODEL_PATH

    if not Path(model_path).exists():
        return (
            f"Model file not found at {model_path}.  "
            "Update YOLO_MODEL_PATH in application/config.py."
        )
    try:
        _models = load_models(model_path, use_gpu=False)
        classes = ", ".join(_models.class_names)
        return f"Models ready. Detecting: {classes}"
    except Exception as e:
        return f"Failed to load models: {e}"


# ── Video source resolution ───────────────────────────────────────────────────


def _resolve_video(video_file, youtube_url: str) -> tuple[str | None, str | None, str | None]:
    if video_file is not None:
        path = video_file if isinstance(video_file, str) else getattr(video_file, "name", None)
        if path:
            return path, None, None

    url = (youtube_url or "").strip()
    if not url:
        return None, None, "Please upload a video file or paste a YouTube URL."

    if not is_youtube_url(url):
        return None, None, "The URL does not look like a YouTube link. Check the format and try again."

    try:
        path, tmp_dir = download_youtube_video(url)
        return path, tmp_dir, None
    except RuntimeError as e:
        return None, None, str(e)


# ── Output helpers ────────────────────────────────────────────────────────────
#
# Every yield in _handle_analyse must have exactly 9 values — one per output
# listed in run_btn.click(outputs=[...]). gr.update() means "no change".


def _empty_outputs(status: str) -> tuple:
    return (
        pd.DataFrame(),           # results_state
        status,                   # status_md
        gr.update(visible=False), # results_section
        pd.DataFrame(),           # results_table
        pd.DataFrame(),           # station_summary_table
        None,                     # timeline_chart
        None,                     # heatmap_chart
        None,                     # pct_chart
        None,                     # volume_chart
        None,                     # pie_chart
    )


# ── Analysis handler (generator) ─────────────────────────────────────────────


def _handle_analyse(video_file, youtube_url: str, target_fps: float, chart_type: str):
    """Generator handler for the Run Analysis button.

    Yields status-only updates during frame processing, then a full result
    on the final yield once post-processing and chart building are done."""

    if _models is None:
        yield _empty_outputs("Models are not loaded. Please restart the application.")
        return

    # Immediate feedback — this yield arrives at the browser before any blocking work starts
    url = (youtube_url or "").strip()
    is_yt = video_file is None and bool(url)
    yield _empty_outputs("Downloading video from YouTube…" if is_yt else "Loading video…")

    video_path, tmp_dir, error = _resolve_video(video_file, youtube_url)
    if error:
        yield _empty_outputs(error)
        return

    yield _empty_outputs("Preparing video…")

    try:
        result_df = pd.DataFrame()

        for update in iter_pipeline(video_path, _models, target_fps):

            if isinstance(update, tuple):
                frame_idx, total = update
                pct = int(frame_idx / total * 100)
                status = f"Processing frame {frame_idx} / {total}  ({pct}%)"
                yield (
                    pd.DataFrame(), status, gr.update(visible=False),
                    pd.DataFrame(), pd.DataFrame(), None, None, None, None, None,
                )

            else:
                result_df = update

        if result_df.empty:
            yield _empty_outputs(
                "No detections found in this video. "
                "Try a lower frame rate or confirm the video shows the train simulator UI."
            )
            return

        yield (
            pd.DataFrame(), "Building charts…", gr.update(visible=False),
            pd.DataFrame(), pd.DataFrame(), None, None, None, None, None,
        )

        station_agg = aggregate_by_station(result_df)
        time_agg    = aggregate_by_time_interval(result_df, interval_minutes=5)
        ts_agg      = aggregate_time_series(result_df)
        hm_agg      = aggregate_heatmap(result_df)

        n_events   = len(result_df)
        n_stations = result_df[
            result_df["station_id"] != "OCR MISSREAD"
        ]["station_name"].nunique()

        yield (
            result_df,
            f"Analysis complete — {n_events} events across {n_stations} stations.",
            gr.update(visible=True),
            result_df,
            station_agg,
            passenger_timeline_chart(ts_agg, chart_type),
            passenger_heatmap_chart(hm_agg),
            passenger_pct_stacked_bar(time_agg),
            passenger_volume_bar(station_agg),
            station_share_pie(station_agg),
        )

    except Exception as e:
        yield _empty_outputs(f"Analysis failed: {e}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Chart refresh handler ─────────────────────────────────────────────────────


def _handle_refresh_charts(df: pd.DataFrame, interval: int, chart_type: str) -> tuple:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(), None, None, None, None, None

    station_agg = aggregate_by_station(df)
    time_agg    = aggregate_by_time_interval(df, interval)
    ts_agg      = aggregate_time_series(df)
    hm_agg      = aggregate_heatmap(df)
    return (
        station_agg,
        passenger_timeline_chart(ts_agg, chart_type),
        passenger_heatmap_chart(hm_agg),
        passenger_pct_stacked_bar(time_agg),
        passenger_volume_bar(station_agg),
        station_share_pie(station_agg),
    )


# ── Download handlers ─────────────────────────────────────────────────────────


def _handle_csv_download(df: pd.DataFrame):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return gr.update(visible=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="train_traffic_")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


def _handle_chart_download(df: pd.DataFrame, interval: int):
    """Renders the destination-share chart to a PNG the user can download.

    Raises gr.Error so the message is shown in the UI if rendering fails
    (previously the handler returned silently and the button looked broken)."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        raise gr.Error("No results to export yet. Run an analysis first.")
    try:
        time_agg = aggregate_by_time_interval(df, interval)
        fig = passenger_pct_stacked_bar(time_agg)
        img_bytes = figure_to_bytes(fig, fmt="png")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="train_traffic_chart_")
        tmp.write(img_bytes)
        tmp.close()
        return gr.update(value=tmp.name, visible=True)
    except RuntimeError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Could not export chart: {e}")


# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    button_primary_background_fill="#15803D",
    button_primary_background_fill_hover="#166534",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#F0FDF4",
    button_secondary_text_color="#15803D",
    button_secondary_border_color="#16A34A",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Design tokens — light mode (default) ─────────────────────────────────── */
:root {
  --bg:          #EFF4F9;
  --surface:     #FFFFFF;
  --surface-hi:  #F8FAFC;
  --border:      #DDE6EF;
  --border-hi:   #B6C8D8;
  --row-divide:  #F1F5F9;
  --ink:         #0F172A;
  --ink-2:       #1E293B;
  --ink-3:       #475569;
  --ink-4:       #64748B;
  --green:       #16A34A;
  --green-dark:  #14532D;
  --green-mid:   #15803D;
  --green-hi:    #15803D;
  --green-bg:    #F0FDF4;
  --shadow-sm:   0 1px 2px rgba(15,23,42,.04), 0 1px 4px rgba(15,23,42,.07);
  --shadow-md:   0 4px 12px -2px rgba(15,23,42,.10), 0 2px 4px -1px rgba(15,23,42,.06);
  --btn-glow:    rgba(22,163,74,.32);
  --btn-glow-hi: rgba(22,163,74,.42);
  --r: 8px;

  /* ─ Pipe tokens into Gradio's own CSS variables so its internal components
     (dropdowns, radios, labels, textboxes, accordions) follow the same palette
     and flip with the OS theme in lock-step with our custom chrome. ─ */
  --body-text-color:            var(--ink-2);
  --body-text-color-subdued:    var(--ink-3);
  --body-background-fill:       var(--bg);
  --background-fill-primary:    var(--surface);
  --background-fill-secondary:  var(--surface-hi);
  --border-color-primary:       var(--border);
  --border-color-accent:        var(--green);
  --block-background-fill:      var(--surface);
  --block-border-color:         var(--border);
  --block-label-text-color:     var(--ink-3);
  --block-title-text-color:     var(--ink-2);
  --input-background-fill:      var(--surface);
  --input-background-fill-focus:var(--surface);
  --input-text-color:           var(--ink);
  --input-border-color:         var(--border);
  --input-border-color-focus:   var(--green);
  --input-placeholder-color:    var(--ink-4);
  --panel-background-fill:      var(--surface);
  --panel-border-color:         var(--border);
  --table-text-color:           var(--ink-2);
  --table-background-fill:      var(--surface);
  --table-border-color:         var(--border);
  --neutral-50:  #F8FAFC;
  --neutral-100: #F1F5F9;
  --neutral-200: #E2E8F0;
  --neutral-300: #CBD5E1;
  --neutral-700: #334155;
  --neutral-800: #1E293B;
  --neutral-900: #0F172A;
}

/* ── Design tokens — dark mode (follows OS/system) ────────────────────────── */
@media (prefers-color-scheme: dark) {
  :root {
    --bg:          #0B1220;
    --surface:     #141E33;
    --surface-hi:  #1C2740;
    --border:      #26354F;
    --border-hi:   #3A4B6A;
    --row-divide:  rgba(255,255,255,.05);
    --ink:         #F8FAFC;
    --ink-2:       #E2E8F0;
    --ink-3:       #CBD5E1;
    --ink-4:       #94A3B8;
    --green:       #22C55E;
    --green-dark:  #14532D;
    --green-mid:   #16A34A;
    --green-hi:    #4ADE80;
    --green-bg:    rgba(34,197,94,.10);
    --shadow-sm:   0 1px 2px rgba(0,0,0,.35), 0 1px 4px rgba(0,0,0,.45);
    --shadow-md:   0 4px 14px rgba(0,0,0,.50), 0 2px 4px rgba(0,0,0,.35);
    --btn-glow:    rgba(34,197,94,.28);
    --btn-glow-hi: rgba(34,197,94,.42);
    --neutral-50:  #1C2740;
    --neutral-100: #26354F;
    --neutral-200: #3A4B6A;
    --neutral-300: #475569;
    --neutral-700: #CBD5E1;
    --neutral-800: #E2E8F0;
    --neutral-900: #F1F5F9;
  }
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
.gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  max-width: 1280px !important;
  color: var(--ink-2) !important;
}
footer { display: none !important; }

/* Bump body text weight across the app for better perceived contrast */
.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container .prose {
  font-weight: 500 !important;
}

/* ── Header ───────────────────────────────────────────────────────────────── */
#app-header {
  background: linear-gradient(140deg, #0F3B22 0%, #15803D 55%, #16A34A 100%);
  border-radius: var(--r);
  padding: 40px 36px 32px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-md);
}
#app-title,
#app-title p {
  font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
  font-size: 44px !important;
  font-weight: 700 !important;
  color: #ffffff !important;
  letter-spacing: -1.2px;
  margin: 0 !important; padding: 0 !important;
  line-height: 1.05 !important;
}
#app-subtitle,
#app-subtitle p {
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  color: rgba(255,255,255,.82) !important;
  margin: 10px 0 0 !important; padding: 0 !important;
  line-height: 1.5 !important;
  max-width: 56ch;
}

/* ── Model status badge ───────────────────────────────────────────────────── */
#model-status {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--green) !important;
  border-radius: var(--r) !important;
  padding: 11px 16px !important;
  box-shadow: var(--shadow-sm) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--ink-2) !important;
  margin-bottom: 0 !important;
}
#model-status p { margin: 0 !important; font-weight: 600 !important; }

/* ── Input card ───────────────────────────────────────────────────────────── */
#input-card {
  background: var(--surface) !important;
  color: var(--ink-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  box-shadow: var(--shadow-sm) !important;
  padding: 20px 20px 16px !important;
}
#input-card label,
#input-card label span,
#input-card .label-wrap > span,
#input-card .block-label {
  color: var(--ink-2) !important;
  font-weight: 600 !important;
}
#input-card input,
#input-card textarea {
  color: var(--ink) !important;
  background: var(--surface) !important;
}

/* ── Section label ────────────────────────────────────────────────────────── */
.section-label,
.section-label p {
  font-size: 14px !important;
  font-weight: 700 !important;
  color: var(--ink-2) !important;
  margin-bottom: 10px !important;
  text-transform: none !important;
  letter-spacing: -0.1px !important;
}

/* ── Run button ───────────────────────────────────────────────────────────── */
#run-btn { width: 100%; }
#run-btn button {
  background: linear-gradient(135deg, var(--green-mid) 0%, var(--green) 100%) !important;
  border: none !important;
  border-radius: 7px !important;
  height: 48px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
  color: #ffffff !important;
  box-shadow: 0 2px 8px var(--btn-glow) !important;
  transition: transform 0.14s ease, box-shadow 0.14s ease !important;
  width: 100% !important;
}
#run-btn button:hover:not(:disabled) {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 18px var(--btn-glow-hi) !important;
}
#run-btn button:active:not(:disabled) {
  transform: translateY(0) !important;
}

/* ── Status line ──────────────────────────────────────────────────────────── */
#status-line {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  color: var(--ink-2) !important;
  min-height: 38px;
  padding: 9px 14px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--border-hi) !important;
  border-radius: 6px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: border-left-color 0.25s ease !important;
}
#status-line p { margin: 0 !important; font-weight: 600 !important; }

#status-line.processing {
  border-left-color: var(--green) !important;
  animation: pulse-accent 2s ease-in-out infinite;
}
@keyframes pulse-accent {
  0%, 100% { border-left-color: var(--green); }
  50%       { border-left-color: rgba(34,197,94,.35); }
}

/* ── Results heading ──────────────────────────────────────────────────────── */
#results-heading,
#results-heading p {
  font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.6px;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 14px !important;
  margin-bottom: 0 !important;
  line-height: 1.1 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.tab-nav button {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: var(--ink-4) !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 18px !important;
  background: transparent !important;
  transition: color 0.12s ease !important;
}
.tab-nav button:hover { color: var(--ink-2) !important; }
.tab-nav button.selected {
  color: var(--green-hi) !important;
  border-bottom-color: var(--green) !important;
  font-weight: 700 !important;
}

/* ── Chart controls row ───────────────────────────────────────────────────── */
#chart-controls {
  background: var(--surface-hi) !important;
  color: var(--ink-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 14px 16px !important;
  margin-bottom: 4px !important;
  align-items: flex-end !important;
}
#chart-controls label,
#chart-controls label span,
#chart-controls .block-label,
#chart-controls .label-wrap > span {
  color: var(--ink-2) !important;
  font-weight: 600 !important;
}

/* ── Catch-all: ensure Gradio form text respects the active token palette ─ */
.gradio-container label,
.gradio-container .label-wrap > span,
.gradio-container .block-label {
  color: var(--ink-2) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ink) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--ink-4) !important;
  opacity: 1;
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
.gradio-dataframe table { font-size: 13px !important; }
.gradio-dataframe thead th {
  background: var(--surface-hi) !important;
  font-weight: 700 !important;
  font-size: 12px !important;
  color: var(--ink-2) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 10px 12px !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}
.gradio-dataframe tbody td {
  padding: 9px 12px !important;
  color: var(--ink-2) !important;
  font-weight: 500 !important;
  border-bottom: 1px solid var(--row-divide) !important;
}
.gradio-dataframe tbody tr:hover td {
  background: var(--green-bg) !important;
}

/* ── Download buttons ─────────────────────────────────────────────────────── */
#dl-csv button, #dl-chart button {
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  height: 38px !important;
  border: 1.5px solid var(--green) !important;
  color: var(--green-hi) !important;
  background: transparent !important;
  transition: background 0.12s ease !important;
}
#dl-csv button:hover, #dl-chart button:hover {
  background: var(--green-bg) !important;
}

/* ── Accordion ────────────────────────────────────────────────────────────── */
.gradio-accordion {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  overflow: hidden;
}

/* ── Plotly charts: dark-mode SVG overrides ───────────────────────────────── */
/* Plotly bakes colors into SVG at render time, so we re-tint them with CSS
   in dark mode to keep text and axes readable on the dark surface. */
@media (prefers-color-scheme: dark) {
  .gradio-container .js-plotly-plot .gtitle { fill: #F1F5F9 !important; }
  .gradio-container .js-plotly-plot text    { fill: #CBD5E1 !important; }
  .gradio-container .js-plotly-plot .legend text,
  .gradio-container .js-plotly-plot .legendtitletext { fill: #CBD5E1 !important; }
  .gradio-container .js-plotly-plot .xaxislayer-above path.domain,
  .gradio-container .js-plotly-plot .yaxislayer-above path.domain,
  .gradio-container .js-plotly-plot .xaxislayer-above .xtick > path,
  .gradio-container .js-plotly-plot .yaxislayer-above .ytick > path {
    stroke: rgba(203,213,225,.35) !important;
  }
  .gradio-container .js-plotly-plot .gridlayer path { stroke: rgba(148,163,184,.12) !important; }
}
"""

# ── JS (runs immediately on button click, before server responds) ─────────────
#
# Must return the inputs verbatim — if it returns nothing/undefined, Gradio 6
# passes `None` for every input to the Python function, breaking the slider.

_ANALYSE_JS = """
(video_file, youtube_url, target_fps, chart_type) => {
    const sline = document.getElementById('status-line');
    if (sline) {
        sline.classList.add('processing');
        const p = sline.querySelector('p');
        if (p) p.textContent = 'Starting…';
    }
    return [video_file, youtube_url, target_fps, chart_type];
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Train Traffic Capture") as app:

    results_state = gr.State(pd.DataFrame())

    # Header
    with gr.Column(elem_id="app-header"):
        gr.Markdown("Train Traffic Capture", elem_id="app-title")
        gr.Markdown(
            "Extract and visualise passenger movement data from train simulator recordings.",
            elem_id="app-subtitle",
        )

    with gr.Column():
        model_status_md = gr.Markdown("Loading models…", elem_id="model-status")

    gr.HTML("<div style='height:16px'></div>")

    # Input
    with gr.Column(elem_id="input-card"):
        gr.Markdown("Video source", elem_classes="section-label")
        with gr.Tabs():
            with gr.Tab("Local file"):
                video_file_input = gr.File(
                    label="Video file",
                    file_types=[".mp4", ".mov", ".webm", ".avi"],
                    file_count="single",
                )
            with gr.Tab("YouTube URL"):
                youtube_url_input = gr.Textbox(
                    placeholder="https://www.youtube.com/watch?v=…",
                    label="YouTube URL",
                    info="yt-dlp downloads the video before processing.",
                )

        with gr.Accordion("Settings", open=False):
            fps_slider = gr.Slider(
                minimum=0.5,
                maximum=3.0,
                value=config.DEFAULT_TARGET_FPS,
                step=0.5,
                label="Sampling rate (frames per second)",
                info="Lower values process faster. 1 fps is a good starting point.",
            )

    gr.HTML("<div style='height:8px'></div>")

    run_btn = gr.Button("Run Analysis", variant="primary", elem_id="run-btn")
    status_md = gr.Markdown("", elem_id="status-line")

    gr.HTML("<div style='height:16px'></div>")

    # Results (hidden until analysis completes)
    with gr.Column(visible=False) as results_section:

        gr.Markdown("Results", elem_id="results-heading")
        gr.HTML("<div style='height:10px'></div>")

        with gr.Tabs():

            with gr.Tab("Raw data"):
                gr.Markdown(
                    "Every captured passenger event. "
                    "Rows labelled OCR MISSREAD had an unreadable station ID.",
                    elem_classes="section-label",
                )
                results_table = gr.Dataframe(wrap=True)

            with gr.Tab("Station summary"):
                gr.Markdown(
                    "Total passenger increases per station across the full recording.",
                    elem_classes="section-label",
                )
                station_summary_table = gr.Dataframe()

            with gr.Tab("Charts"):
                # Controls
                with gr.Row(elem_id="chart-controls"):
                    interval_dropdown = gr.Dropdown(
                        choices=[1, 5, 10, 15, 30],
                        value=5,
                        label="Destination share — interval (minutes)",
                        scale=3,
                    )
                    chart_type_radio = gr.Radio(
                        choices=["bar", "line"],
                        value="bar",
                        label="Timeline chart type",
                        scale=2,
                    )
                    refresh_btn = gr.Button("Refresh charts", size="sm", scale=1)

                gr.Markdown(
                    "Passenger increases over in-game time, broken out by destination station.",
                    elem_classes="section-label",
                )
                timeline_chart = gr.Plot()

                gr.HTML("<div style='height:12px'></div>")

                gr.Markdown(
                    "Crowd heatmap — where and when passengers concentrate. "
                    "Darker cells mean higher passenger volume at that station and in-game minute.",
                    elem_classes="section-label",
                )
                heatmap_chart = gr.Plot()

                gr.HTML("<div style='height:12px'></div>")

                gr.Markdown(
                    "Destination share by time interval — each bar shows what percentage "
                    "of passenger increases went to each station within that window.",
                    elem_classes="section-label",
                )
                pct_chart = gr.Plot()

                with gr.Accordion("More charts", open=False):
                    volume_chart = gr.Plot(label="Total passengers per station")
                    pie_chart    = gr.Plot(label="Station share — full recording")

            with gr.Tab("Export"):
                gr.Markdown(
                    "Download the captured data or the chart to your computer.",
                    elem_classes="section-label",
                )
                with gr.Row():
                    csv_btn   = gr.Button("Download CSV",         variant="secondary", elem_id="dl-csv")
                    chart_btn = gr.Button("Download chart (PNG)", variant="secondary",
                                          elem_id="dl-chart", visible=_PNG_EXPORT_ENABLED)
                csv_file_output   = gr.File(label="CSV file",           visible=False)
                chart_file_output = gr.File(label="Chart image (PNG)",  visible=False)

    # ── Event wiring ──────────────────────────────────────────────────────────

    run_btn.click(
        fn=_handle_analyse,
        inputs=[video_file_input, youtube_url_input, fps_slider, chart_type_radio],
        outputs=[
            results_state,
            status_md,
            results_section,
            results_table,
            station_summary_table,
            timeline_chart,
            heatmap_chart,
            pct_chart,
            volume_chart,
            pie_chart,
        ],
        js=_ANALYSE_JS,
    )

    refresh_btn.click(
        fn=_handle_refresh_charts,
        inputs=[results_state, interval_dropdown, chart_type_radio],
        outputs=[
            station_summary_table,
            timeline_chart,
            heatmap_chart,
            pct_chart,
            volume_chart,
            pie_chart,
        ],
    )

    csv_btn.click(
        fn=_handle_csv_download,
        inputs=[results_state],
        outputs=[csv_file_output],
    )

    chart_btn.click(
        fn=_handle_chart_download,
        inputs=[results_state, interval_dropdown],
        outputs=[chart_file_output],
    )

    app.load(fn=_warmup, outputs=[model_status_md])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch(show_error=True, theme=_THEME, css=_CSS)
