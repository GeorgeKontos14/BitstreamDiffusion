from __future__ import annotations
 
import html as _html
import json
import math
from pathlib import Path
from typing import Any, Dict

TASK_META = {
    "joint": {
        "name": "Joint Generation",
        "description": (
            "Generate both text and speech simultaneously from scratch. "
            "The model produces the full sequence without any conditioning "
            "input, demonstrating its ability to jointly model both modalities."
        ),
        "metrics": {
            "wer": {
                "name": "Word Error Rate",
                "detail": (
                    "Generated audio is transcribed with Whisper and compared "
                    "to the generated text. Lower is better (0 = perfect)."
                ),
            },
            "utmos": {
                "name": "UTMOS",
                "detail": (
                    "Predicted Mean Opinion Score for speech quality "
                    "(1–5 scale). Higher indicates better perceived quality."
                ),
            },
        },
        "columns": [
            ("gen_text", "Generated Text", "text"),
            ("gen_wav", "Generated Audio", "audio"),
            ("whisper", "Whisper Transcription", "text"),
        ],
    },
    "tts": {
        "name": "Text‑to‑Speech",
        "description": (
            "Given reference text, generate corresponding speech audio. "
            "Evaluates the model’s ability to produce intelligible and "
            "natural‑sounding speech from text input."
        ),
        "metrics": {
            "wer": {
                "name": "Word Error Rate",
                "detail": (
                    "Generated audio is transcribed with Whisper and compared "
                    "to the reference text. Lower is better."
                ),
            },
            "utmos": {
                "name": "UTMOS",
                "detail": (
                    "Predicted Mean Opinion Score for speech quality "
                    "(1–5 scale). Higher is better."
                ),
            },
        },
        "columns": [
            ("ref_text", "Reference Text", "text"),
            ("ref_wav", "Reference Audio", "audio"),
            ("gen_wav", "Generated Audio", "audio"),
            ("whisper", "Whisper Transcription", "text"),
        ],
    },
    "stt": {
        "name": "Speech‑to‑Text",
        "description": (
            "Given reference speech audio, generate the corresponding text. "
            "Evaluates the model’s ability to transcribe speech without "
            "an explicit ASR module."
        ),
        "metrics": {
            "wer": {
                "name": "Word Error Rate",
                "detail": (
                    "Generated text is compared to the reference text. "
                    "Lower is better (0 = perfect transcription)."
                ),
            },
        },
        "columns": [
            ("ref_wav", "Reference Audio", "audio"),
            ("ref_text", "Reference Text", "text"),
            ("gen_text", "Generated Text", "text"),
        ],
    },
    "cont": {
        "name": "Continuation",
        "description": (
            "Given a prefix of text and speech, continue generating both "
            "modalities. Evaluates the model’s ability to produce "
            "coherent continuations across text and audio."
        ),
        "metrics": {
            "utmos": {
                "name": "UTMOS",
                "detail": (
                    "Predicted Mean Opinion Score for the quality of "
                    "continued speech (1–5 scale). Higher is better."
                ),
            },
        },
        "columns": [
            ("gen_text", "Generated Text", "text"),
            ("gen_wav", "Generated Audio", "audio"),
        ],
    },
}
 
TASK_ORDER = ["joint", "tts", "stt", "cont"]
 
# ── CSS ─────────────────────────────────────────────────────────────────────
 
_CSS = """\
:root {
  --bg: #ffffff;
  --bg-alt: #f7f8fa;
  --text: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --accent-bg: #eff6ff;
  --radius: 8px;
  --shadow: 0 1px 3px 0 rgba(0,0,0,0.08);
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
  --mono: "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;
}
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 56px; }
body {
  font-family: var(--font); color: var(--text);
  background: var(--bg); line-height: 1.6; margin: 0; padding: 0;
}
 
/* ── Navigation ──────────────────────────────────────────── */
.topnav {
  position: sticky; top: 0; z-index: 100;
  background: var(--bg); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 1rem;
  padding: 0 2rem; height: 52px;
  box-shadow: var(--shadow);
}
.nav-title {
  font-weight: 700; font-size: 0.9rem; color: var(--text-muted);
  white-space: nowrap; margin-right: auto;
}
.nav-links { display: flex; gap: 0.25rem; }
.nav-links a {
  color: var(--accent); text-decoration: none; font-weight: 500;
  font-size: 0.875rem; padding: 0.35rem 0.75rem;
  border-radius: 4px; transition: background 0.15s;
}
.nav-links a:hover { background: var(--accent-bg); }
 
/* ── Container ───────────────────────────────────────────── */
.container { max-width: 1440px; margin: 0 auto; padding: 2rem; }
 
/* ── Header ──────────────────────────────────────────────── */
header h1 { font-size: 1.75rem; margin-bottom: 1.25rem; }
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 0.5rem 1.5rem; margin-bottom: 2.5rem;
  padding: 1.25rem; background: var(--bg-alt);
  border: 1px solid var(--border); border-radius: var(--radius);
}
.meta-item { display: flex; gap: 0.5rem; font-size: 0.875rem; }
.meta-label {
  font-weight: 600; color: var(--text-muted);
  white-space: nowrap; min-width: 120px;
}
.meta-value { color: var(--text); word-break: break-word; }
 
/* ── Task sections ───────────────────────────────────────── */
section {
  margin-bottom: 3rem; padding-top: 0.5rem;
  border-top: 2px solid var(--accent);
}
section h2 { font-size: 1.35rem; margin-bottom: 0.5rem; }
.task-desc {
  color: var(--text-muted); font-size: 0.9rem;
  max-width: 80ch; margin-bottom: 1rem;
}
 
/* ── Metric cards ────────────────────────────────────────── */
.metrics-grid {
  display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;
}
.metric-card {
  background: var(--bg-alt); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.25rem;
  min-width: 220px; flex: 1; max-width: 360px;
}
.metric-value {
  font-size: 2rem; font-weight: 700; color: var(--accent);
  font-family: var(--mono);
}
.metric-name {
  font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted); margin-top: 0.15rem;
}
.metric-detail {
  font-size: 0.8rem; color: var(--text-muted);
  margin-top: 0.4rem; line-height: 1.45;
}
 
/* ── Sample table ────────────────────────────────────────── */
.table-wrap { overflow-x: auto; }
.samples-table {
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
.samples-table th {
  background: var(--bg-alt); border: 1px solid var(--border);
  padding: 0.6rem 0.75rem; text-align: left; font-weight: 600;
  white-space: nowrap; z-index: 10;
}
.samples-table td {
  border: 1px solid var(--border); padding: 0.6rem 0.75rem;
  vertical-align: top;
}
.samples-table tbody tr:nth-child(even) { background: var(--bg-alt); }
.samples-table tbody tr:hover { background: var(--accent-bg); }
 
.idx-cell {
  text-align: center; font-family: var(--mono);
  font-size: 0.8rem; color: var(--text-muted); width: 3em;
}
.text-cell {
  max-width: 400px; word-wrap: break-word;
  white-space: normal; line-height: 1.5;
  max-height: 8em; overflow-y: auto;
}
.audio-cell { white-space: nowrap; }
.audio-cell audio { width: 220px; height: 36px; }
td.empty { color: var(--text-muted); font-style: italic; }
 
.no-data {
  color: var(--text-muted); font-style: italic;
  padding: 1rem 0;
}
 
/* ── Footer ──────────────────────────────────────────────── */
.footer {
  margin-top: 3rem; padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.8rem; color: var(--text-muted); text-align: center;
}
"""
 
# ── HTML rendering helpers ──────────────────────────────────────────────────
 
def _esc(text: Any) -> str:
    return _html.escape(str(text)) if text is not None else ""
 
 
def _fmt_metric(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.4f}"
 
 
def _render_meta_item(label: str, value: str) -> str:
    return (
        f'<div class="meta-item">'
        f'<span class="meta-label">{_esc(label)}</span>'
        f'<span class="meta-value">{_esc(value)}</span>'
        f"</div>"
    )
 
 
def _render_metric_card(key: str, value, task_metrics_meta: dict) -> str:
    info = task_metrics_meta.get(key, {})
    name = info.get("name", key.upper())
    detail = info.get("detail", "")
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{_fmt_metric(value)}</div>'
        f'<div class="metric-name">{_esc(name)}</div>'
        f'<div class="metric-detail">{_esc(detail)}</div>'
        f"</div>"
    )
 
 
def _render_cell(value, col_type: str) -> str:
    if value is None:
        return '<td class="text-cell empty">—</td>'
    if col_type == "audio":
        src = _esc(str(value))
        return (
            f'<td class="audio-cell">'
            f'<audio controls preload="metadata">'
            f'<source src="{src}" type="audio/wav">'
            f"</audio></td>"
        )
    escaped = _html.escape(str(value))
    escaped = escaped.replace('\n', '<br>').replace('\r', '').replace('\t', ' ')
    return f'<td class="text-cell">{escaped}</td>'
 
 
def _render_task_section(task_key: str, task_data: dict, task_idx: int) -> str:
    meta = TASK_META.get(task_key, {})
    metrics = task_data.get("metrics", {})
    samples = task_data.get("samples", [])
 
    name = meta.get("name", task_key)
    desc = meta.get("description", "")
    metrics_meta = meta.get("metrics", {})
    columns = meta.get("columns", [])
 
    # Metric cards
    cards = ""
    for mk in metrics_meta:
        if mk in metrics:
            cards += _render_metric_card(mk, metrics[mk], metrics_meta)
 
    # Table header
    th = "<th>#</th>" + "".join(
        f"<th>{_esc(col_label)}</th>" for _, col_label, _ in columns
    )
 
    # Table rows
    rows = ""
    for sample in samples:
        cells = f'<td class="idx-cell">{sample.get("idx", "")}</td>'
        for col_key, _, col_type in columns:
            cells += _render_cell(sample.get(col_key), col_type)
        rows += f"<tr>{cells}</tr>\n"
 
    no_data = ""
    if not samples:
        no_data = '<p class="no-data">No samples generated for this task.</p>'
 
    table_html = ""
    if samples:
        table_html = (
            f'<div class="table-wrap">\n'
            f'<table class="samples-table">\n'
            f"<thead><tr>{th}</tr></thead>\n"
            f"<tbody>\n{rows}</tbody>\n"
            f"</table>\n</div>"
        )
 
    return (
        f'<section id="{task_key}">\n'
        f"<h2>{task_idx + 1}. {_esc(name)}</h2>\n"
        f'<p class="task-desc">{_esc(desc)}</p>\n'
        f'<div class="metrics-grid">{cards}</div>\n'
        f"{no_data}{table_html}\n"
        f"</section>\n"
    )
 
 
def _render_html(data: dict) -> str:
    header = data.get("header", {})
    tasks_data = data.get("tasks", {})
 
    # ── Sampler info ────────────────────────────────────────
    sampler_raw = header.get("sampler", {})
    if isinstance(sampler_raw, dict):
        sampler_name = sampler_raw.get("name", "unknown")
        num_steps = sampler_raw.get("num_steps", "?")
        terminal_sigma = sampler_raw.get("terminal_sigma", "?")
    else:
        sampler_name = str(sampler_raw)
        num_steps = header.get("num_steps", "?")
        terminal_sigma = header.get("terminal_sigma", "?")
 
    # ── Sequence layout ─────────────────────────────────────
    seq = header.get("sequence_layout", {})
    if isinstance(seq, dict):
        layout_str = (
            f"{seq.get('text_tokens', '?')} text + "
            f"{seq.get('speaker_tokens', '?')} speaker + "
            f"{seq.get('speech_tokens', '?')} speech = "
            f"{seq.get('total_tokens', '?')} tokens"
        )
    else:
        layout_str = str(seq)
 
    # ── Meta items ──────────────────────────────────────────
    step = header.get("step", "?")
    meta_items = "".join([
        _render_meta_item("Text Tokenizer", header.get("text_tokenizer", "unknown")),
        _render_meta_item("Speaker Tokenizer", header.get("speaker_tokenizer", "unknown")),
        _render_meta_item(
            "Speech Tokenizer",
            f"{header.get('speech_tokenizer', 'unknown')} "
            f"({header.get('speech_tokenizer_bottleneck', '')})",
        ),
        _render_meta_item("Bits / Token", header.get("bits_per_token", "?")),
        _render_meta_item("Sequence Layout", layout_str),
        _render_meta_item("Validation Split", header.get("split", "val")),
        _render_meta_item(
            "Checkpoint",
            f"Step {step} (Epoch {header.get('epoch', '?')})",
        ),
        _render_meta_item("Samples / Task", header.get("num_samples", "?")),
        _render_meta_item(
            "Sampler",
            f"{sampler_name} · {num_steps} steps · "
            f"σ_min = {terminal_sigma}",
        ),
    ])
 
    # ── Nav links ───────────────────────────────────────────
    nav_links = ""
    for tk in TASK_ORDER:
        if tk in tasks_data:
            name = TASK_META.get(tk, {}).get("name", tk)
            nav_links += f'<a href="#{tk}">{_esc(name)}</a>\n'
 
    # ── Task sections ───────────────────────────────────────
    sections = ""
    idx = 0
    for tk in TASK_ORDER:
        if tk in tasks_data:
            sections += _render_task_section(tk, tasks_data[tk], idx)
            idx += 1
 
    title = _esc(header.get("title", "TextAudio Evaluation"))
 
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Step {_esc(str(step))}</title>
<style>
{_CSS}
</style>
</head>
<body>
<nav class="topnav">
  <span class="nav-title">TextAudio Eval · Step {_esc(str(step))}</span>
  <div class="nav-links">{nav_links}</div>
</nav>
<div class="container">
  <header>
    <h1>{title}</h1>
    <div class="meta-grid">{meta_items}</div>
  </header>
  {sections}
  <div class="footer">
    Generated by TextAudioCallback
  </div>
</div>
</body>
</html>"""
 
 
# ── Public API ──────────────────────────────────────────────────────────────
 
def build_textaudio_report(save_dir) -> Path:
    """
    Read data.json from *save_dir* and write report.html next to it.
 
    Parameters
    ----------
    save_dir : str or Path
        Directory containing data.json and task sub-folders with .wav files.
 
    Returns
    -------
    Path to the generated report.html.
    """
    save_dir = Path(save_dir)
    data_path = save_dir / "data.json"
 
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    html_str = _render_html(data)
    out_path = save_dir / "report.html"
 
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
 
    return out_path