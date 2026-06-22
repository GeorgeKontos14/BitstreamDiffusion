from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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
                    "to the generated text. Lower is better (0 = perfect)."
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
            "Evaluates the model's ability to produce intelligible and "
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
            "Evaluates the model's ability to transcribe speech without "
            "an explicit ASR module."
        ),
        "metrics": {
            "wer": {
                "name": "Word Error Rate",
                "detail": (
                    "Generated text is compared to the reference text. "
                    "Lower is better (0 = perfect transcription)."
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
            "modalities. Evaluates the model's ability to produce "
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

/* ── Metric cards (single-sampler fallback) ──────────────── */
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

/* ── Metrics comparison table (multi-sampler) ────────────── */
.metrics-compare-wrap { margin-bottom: 1.5rem; overflow-x: auto; }
.metrics-compare-table {
  border-collapse: collapse; font-size: 0.875rem;
  min-width: 280px;
}
.metrics-compare-table th {
  background: var(--bg-alt); border: 1px solid var(--border);
  padding: 0.55rem 1rem; text-align: left; font-weight: 600;
  font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted);
}
.metrics-compare-table td {
  border: 1px solid var(--border); padding: 0.55rem 1rem;
}
.metrics-compare-table .sampler-col {
  font-family: var(--mono); font-size: 0.82rem;
  font-weight: 600; color: var(--text);
}
.metrics-compare-table .num-col {
  font-family: var(--mono); color: var(--accent);
  font-weight: 700; text-align: right;
}
.metrics-compare-table tbody tr:nth-child(even) { background: var(--bg-alt); }
.metrics-compare-table tbody tr:hover { background: var(--accent-bg); }

/* ── Sampler tab strip ───────────────────────────────────── */
.tab-strip {
  display: flex; gap: 0; flex-wrap: wrap;
  border-bottom: 2px solid var(--border);
  margin-top: 1.5rem; margin-bottom: 0;
}
.tab-btn {
  padding: 0.45rem 1.1rem;
  border: 1px solid var(--border); border-bottom: none;
  background: var(--bg-alt); cursor: pointer;
  font-size: 0.8rem; font-weight: 500;
  border-radius: 6px 6px 0 0; color: var(--text-muted);
  transition: background 0.12s, color 0.12s;
  font-family: var(--mono); margin-right: 3px; margin-bottom: -2px;
}
.tab-btn:hover { background: var(--accent-bg); color: var(--accent); }
.tab-btn.active {
  background: var(--bg); color: var(--accent);
  border-color: var(--border); border-bottom-color: var(--bg);
  font-weight: 600;
}
.sampler-panel { padding-top: 1rem; }

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

_JS = """\
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var task = this.dataset.task;
    var sampler = this.dataset.sampler;
    document.querySelectorAll('.tab-btn[data-task="' + task + '"]').forEach(function(b) {
      b.classList.remove('active');
    });
    this.classList.add('active');
    document.querySelectorAll('.sampler-panel[data-task="' + task + '"]').forEach(function(p) {
      p.style.display = 'none';
    });
    var active = document.querySelector(
      '.sampler-panel[data-task="' + task + '"][data-sampler="' + sampler + '"]'
    );
    if (active) active.style.display = 'block';
  });
});
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


def _render_metrics_comparison(task_key: str, sampler_results: Dict[str, dict]) -> str:
    """Compact table comparing metrics across all samplers for one task."""
    metrics_meta = TASK_META.get(task_key, {}).get("metrics", {})
    metric_keys = list(metrics_meta.keys())
    if not metric_keys:
        return ""

    th = "<th>Sampler</th>" + "".join(
        f"<th>{_esc(metrics_meta[mk]['name'])}</th>" for mk in metric_keys
    )

    rows = ""
    for sampler_name, task_data in sampler_results.items():
        metrics = task_data.get("metrics", {})
        cells = f'<td class="sampler-col">{_esc(sampler_name)}</td>'
        for mk in metric_keys:
            cells += f'<td class="num-col">{_fmt_metric(metrics.get(mk))}</td>'
        rows += f"<tr>{cells}</tr>\n"

    return (
        f'<div class="metrics-compare-wrap">\n'
        f'<table class="metrics-compare-table">\n'
        f"<thead><tr>{th}</tr></thead>\n"
        f"<tbody>\n{rows}</tbody>\n"
        f"</table>\n</div>"
    )


def _render_sample_table(columns: list, samples: list) -> str:
    if not samples:
        return '<p class="no-data">No samples generated for this task.</p>'

    th = "<th>#</th>" + "".join(
        f"<th>{_esc(col_label)}</th>" for _, col_label, _ in columns
    )
    rows = ""
    for sample in samples:
        cells = f'<td class="idx-cell">{sample.get("idx", "")}</td>'
        for col_key, _, col_type in columns:
            cells += _render_cell(sample.get(col_key), col_type)
        rows += f"<tr>{cells}</tr>\n"

    return (
        f'<div class="table-wrap">\n'
        f'<table class="samples-table">\n'
        f"<thead><tr>{th}</tr></thead>\n"
        f"<tbody>\n{rows}</tbody>\n"
        f"</table>\n</div>"
    )


def _render_task_section(
    task_key: str,
    sampler_results: Dict[str, dict],
    task_idx: int,
) -> str:
    """
    sampler_results: {sampler_name: {metrics: {...}, samples: [...]}}
    Renders one task section. When there are multiple samplers the metrics are
    shown as a comparison table and samples are toggled via tab buttons.
    """
    meta = TASK_META.get(task_key, {})
    name = meta.get("name", task_key)
    desc = meta.get("description", "")
    columns = meta.get("columns", [])
    metrics_meta = meta.get("metrics", {})

    sampler_names = list(sampler_results.keys())
    multi = len(sampler_names) > 1

    # ── Metrics block ────────────────────────────────────────
    if multi:
        metrics_html = _render_metrics_comparison(task_key, sampler_results)
    else:
        single_metrics = (sampler_results[sampler_names[0]] if sampler_names else {}).get("metrics", {})
        cards = "".join(
            _render_metric_card(mk, single_metrics.get(mk), metrics_meta)
            for mk in metrics_meta
            if mk in single_metrics
        )
        metrics_html = f'<div class="metrics-grid">{cards}</div>'

    # ── Tab strip (multi-sampler only) ───────────────────────
    tab_strip = ""
    if multi:
        tabs = "".join(
            f'<button class="tab-btn{" active" if i == 0 else ""}" '
            f'data-task="{_esc(task_key)}" data-sampler="{_esc(sname)}">'
            f'{_esc(sname)}</button>\n'
            for i, sname in enumerate(sampler_names)
        )
        tab_strip = f'<div class="tab-strip">{tabs}</div>\n'

    # ── Sample panels ────────────────────────────────────────
    panels = ""
    for i, (sname, task_data) in enumerate(sampler_results.items()):
        style = '' if i == 0 else ' style="display:none"'
        data_attrs = f'data-task="{_esc(task_key)}" data-sampler="{_esc(sname)}"'
        table_html = _render_sample_table(columns, task_data.get("samples", []))
        panels += (
            f'<div class="sampler-panel" {data_attrs}{style}>\n'
            f'{table_html}\n'
            f'</div>\n'
        )

    return (
        f'<section id="{task_key}">\n'
        f"<h2>{task_idx + 1}. {_esc(name)}</h2>\n"
        f'<p class="task-desc">{_esc(desc)}</p>\n'
        f'{metrics_html}\n'
        f'{tab_strip}'
        f'{panels}'
        f"</section>\n"
    )


# ── Data normalisation (backward compat with old single-sampler format) ──────

def _normalize_data(data: dict) -> dict:
    """Convert old data.json (top-level 'tasks') to new format (top-level 'samplers')."""
    if "samplers" in data:
        return data

    old_header = data.get("header", {})
    sampler_raw = old_header.get("sampler", {})
    if isinstance(sampler_raw, dict):
        sampler_name = sampler_raw.get("name", "default")
        num_steps = sampler_raw.get("num_steps", "?")
        terminal_sigma = sampler_raw.get("terminal_sigma", "?")
    else:
        sampler_name = str(sampler_raw) or "default"
        num_steps = old_header.get("num_steps", "?")
        terminal_sigma = old_header.get("terminal_sigma", "?")

    new_header = {k: v for k, v in old_header.items() if k != "sampler"}
    new_header["samplers"] = [{"name": sampler_name, "num_steps": num_steps, "terminal_sigma": terminal_sigma}]

    return {
        "header": new_header,
        "samplers": {sampler_name: {"tasks": data.get("tasks", {})}},
    }


def _render_html(data: dict) -> str:
    data = _normalize_data(data)
    header = data.get("header", {})
    samplers_data = data.get("samplers", {})  # {sampler_name: {tasks: {task: data}}}

    # Reorganise to {task: {sampler_name: task_data}} for rendering
    task_sampler_results: Dict[str, Dict[str, dict]] = {}
    for sampler_name, sd in samplers_data.items():
        for task_key, task_data in sd.get("tasks", {}).items():
            task_sampler_results.setdefault(task_key, {})[sampler_name] = task_data

    # ── Sampler summary for meta grid ───────────────────────
    samplers_list = header.get("samplers", [])
    if not samplers_list:
        sampler_str = "unknown"
    elif len(samplers_list) == 1:
        s = samplers_list[0]
        sampler_str = (
            f"{s.get('name', '?')} · {s.get('num_steps', '?')} steps"
            f" · σ_min = {s.get('terminal_sigma', '?')}"
        )
    else:
        names = ", ".join(s.get("name", "?") for s in samplers_list)
        s0 = samplers_list[0]
        sampler_str = (
            f"{names} · {s0.get('num_steps', '?')} steps"
            f" · σ_min = {s0.get('terminal_sigma', '?')}"
        )

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
        _render_meta_item("Checkpoint", f"Step {step} (Epoch {header.get('epoch', '?')})"),
        _render_meta_item("Samples / Task", header.get("num_samples", "?")),
        _render_meta_item("Sampler(s)", sampler_str),
    ])

    # ── Nav links ───────────────────────────────────────────
    nav_links = "".join(
        f'<a href="#{tk}">{_esc(TASK_META.get(tk, {}).get("name", tk))}</a>\n'
        for tk in TASK_ORDER
        if tk in task_sampler_results
    )

    # ── Task sections ───────────────────────────────────────
    sections = ""
    idx = 0
    for tk in TASK_ORDER:
        if tk in task_sampler_results:
            sections += _render_task_section(tk, task_sampler_results[tk], idx)
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
<script>
{_JS}
</script>
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
