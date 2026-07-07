from __future__ import annotations

import html as _html
import json
import math
import re
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
        # Keys here must match TextAudioEvaluator.evaluate_task_brief's metrics
        # dict verbatim -- task-specific names (e.g. "Cross-Modal WER") are
        # explained once, right under the task description, from legend.txt's
        # "Task-Specific Metrics:" section (see _render_task_section).
        "metrics": {
            "Cross-Modal WER": {"name": "Cross-Modal WER"},
            "Cross-Modal CER": {"name": "Cross-Modal CER"},
            "UTMOS": {"name": "UTMOS"},
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
            "ASR-WER": {"name": "ASR-WER"},
            "ASR-CER": {"name": "ASR-CER"},
            "UTMOS": {"name": "UTMOS"},
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
            "WER": {"name": "WER"},
            "CER": {"name": "CER"},
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
            "WER": {"name": "WER"},
            "CER": {"name": "CER"},
            "UTMOS": {"name": "UTMOS"},
        },
        "columns": [
            ("gen_text", "Generated Text", "text"),
            ("gen_wav", "Generated Audio", "audio"),
        ],
    },
}

TASK_ORDER = ["joint", "tts", "stt", "cont"]

# True = higher is better, False = lower is better. Keyed by the base metric
# type, inferred from a (possibly task-specific) display name via
# _infer_metric_type -- e.g. "Cross-Modal WER" and "ASR-WER" both infer "wer".
METRIC_DIRECTION: Dict[str, bool] = {
    "wer": False,
    "cer": False,
    "utmos": True,
    "genppl": False,
    "spksim": True,
    "fsd": False,
    "llm-as-judge-mos": True,
    "salmon": True,
}

# Longer/more specific patterns first so e.g. "UTMOS" doesn't fall through to
# a generic "MOS" match intended for "LLM-as-Judge-MOS".
_METRIC_TYPE_PATTERNS = [
    ("WER", "wer"),
    ("CER", "cer"),
    ("UTMOS", "utmos"),
    ("GENPPL", "genppl"),
    ("SPKSIM", "spksim"),
    ("FSD", "fsd"),
    ("SALMON", "salmon"),
    ("JUDGE", "llm-as-judge-mos"),
]


def _infer_metric_type(name: str) -> str | None:
    """Maps a possibly task-specific display name (e.g. "Cross-Modal WER",
    "ASR-CER") to its base metric type (e.g. "wer", "cer") for direction/best
    lookups."""
    upper = str(name).upper()
    for pattern, mtype in _METRIC_TYPE_PATTERNS:
        if pattern in upper:
            return mtype
    return None


def _direction_arrow(metric_key: str) -> str:
    higher_is_better = METRIC_DIRECTION.get(_infer_metric_type(metric_key))
    if higher_is_better is None:
        return ""
    return " ↑" if higher_is_better else " ↓"


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

/* ── Legend ──────────────────────────────────────────────── */
.legend-groups {
  display: flex; flex-wrap: wrap; gap: 2rem; margin-bottom: 1rem;
}
.legend-group { flex: 1; min-width: 280px; }
.legend-group h3 {
  font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted); margin: 0 0 0.75rem;
}
.legend-list { margin: 0; display: grid; grid-template-columns: max-content 1fr; column-gap: 1rem; row-gap: 0.6rem; }
.legend-list dt {
  font-weight: 600; font-family: var(--mono); font-size: 0.82rem;
  white-space: nowrap; color: var(--text);
}
.legend-list dd { margin: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }
.task-specific-metrics { margin-bottom: 1.25rem; }

/* Stacked numerator/denominator fraction, e.g. inside a legend explanation. */
.frac {
  display: inline-flex; flex-direction: column; vertical-align: middle;
  text-align: center; margin: 0 0.25em; font-size: 0.92em; line-height: 1.15;
}
.frac .num { border-bottom: 1px solid currentColor; padding: 0 0.3em 0.1em; }
.frac .den { padding: 0.1em 0.3em 0; }

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
.metrics-compare-table .metric-col {
  font-family: var(--mono); color: var(--text);
  text-align: right;
}
.metrics-compare-table .metric-col.best {
  color: var(--accent); font-weight: 700;
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
/* Fixed-height + scrollable: with ~128 samples/task these get very long. */
.table-wrap { max-height: 32rem; overflow-y: auto; overflow-x: auto; }
.samples-table {
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
.samples-table th {
  background: var(--bg-alt); border: 1px solid var(--border);
  padding: 0.6rem 0.75rem; text-align: left; font-weight: 600;
  white-space: nowrap; position: sticky; top: 0; z-index: 10;
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
    name = info.get("name", key)
    detail = info.get("detail", "")
    detail_html = f'<div class="metric-detail">{_esc(detail)}</div>' if detail else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{_fmt_metric(value)}</div>'
        f'<div class="metric-name">{_esc(name)}{_esc(_direction_arrow(key))}</div>'
        f"{detail_html}"
        f"</div>"
    )


def _render_cell(value, col_type: str) -> str:
    if value is None:
        return '<td class="text-cell empty">—</td>'
    if col_type == "audio":
        src = _esc(str(value))
        return (
            f'<td class="audio-cell">'
            f'<audio controls preload="none">'
            f'<source src="{src}" type="audio/wav">'
            f"</audio></td>"
        )
    escaped = _html.escape(str(value))
    escaped = escaped.replace('\n', '<br>').replace('\r', '').replace('\t', ' ')
    return f'<td class="text-cell">{escaped}</td>'


# ── Legend (assets/report/legend.txt) ────────────────────────────────────────

LEGEND_PATH = Path(__file__).resolve().parent.parent / "assets" / "report" / "legend.txt"

# Only metrics this callback actually computes are shown; the rest of
# legend.txt's "Metrics:" section is for the full evaluation pipeline's report.
LEGEND_METRICS_SHOWN = {"WER", "CER", "UTMOS"}

_TRAILING_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")
# Marks a fraction inline in explanation text, e.g. "{a + b}/{c}".
_FRACTION_RE = re.compile(r"\{([^{}]+)\}\s*/\s*\{([^{}]+)\}")


def _legend_metric_abbrev(name: str) -> str:
    """"Word Error Rate (WER)" -> "WER"; "UTMOS" -> "UTMOS" (no parens = use as-is)."""
    m = _TRAILING_PAREN_RE.search(name)
    return (m.group(1) if m else name).strip()


# Recognized top-level sections in legend.txt. A bare "Name:" line is a new
# top-level section only if it's one of these; otherwise, while inside
# "Task-Specific Metrics", it's a task subsection (e.g. "Joint Generation:").
_LEGEND_TOP_SECTIONS = {"Fields", "Metrics", "Task-Specific Metrics"}


def _normalize_label(s: str) -> str:
    """Normalizes a task/section label for matching, tolerating the
    non-breaking hyphen (U+2011) used in TASK_META's task names vs. the plain
    ASCII hyphen a human is likely to type in legend.txt."""
    return s.replace("‑", "-").strip().lower()


def _parse_legend(path: Path) -> Dict[str, Any]:
    """
    Parses legend.txt into:
      {
        "Fields": {name: explanation},
        "Metrics": {name: explanation},
        "Task-Specific Metrics": {task_label: {name: explanation}},
      }

    Format:
      - "Section:" (nothing after the colon) starts a top-level section.
      - Inside "Task-Specific Metrics", a further bare "Label:" line (not
        itself a top-level section name) starts a task subsection.
      - "Name: explanation" is a leaf entry under the current
        section/subsection.
      - A "- " prefixed line is a continuation folded into the previous leaf
        entry (e.g. SALMON's per-judge breakdown) UNLESS it appears directly
        under a task subsection, in which case it's its own leaf entry (e.g.
        "- Cross-Modal WER: ...").
    """
    sections: Dict[str, Any] = {}
    if not path.exists():
        return sections

    current_section: str | None = None
    current_subsection: str | None = None
    current_entry: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("-"):
            bullet_name, _, bullet_rest = line[1:].strip().partition(":")
            bullet_name = bullet_name.strip()
            bullet_rest = bullet_rest.strip()

            if current_section == "Task-Specific Metrics" and current_subsection is not None:
                sections[current_section][current_subsection][bullet_name] = bullet_rest
                continue

            if current_section is not None and current_entry is not None:
                sections[current_section][current_entry] += "; " + line[1:].strip()
                continue

            continue

        if ":" not in line:
            continue

        name, _, rest = line.partition(":")
        name = name.strip()
        rest = rest.strip()

        if not rest:
            if current_section == "Task-Specific Metrics" and name not in _LEGEND_TOP_SECTIONS:
                current_subsection = name
                sections[current_section].setdefault(name, {})
            else:
                current_section = name
                current_subsection = None
                current_entry = None
                sections.setdefault(name, {})
            continue

        if current_section == "Task-Specific Metrics" and current_subsection is not None:
            sections[current_section][current_subsection][name] = rest
        elif current_section is not None:
            sections[current_section][name] = rest
            current_entry = name

    return sections


def _task_specific_metrics_for(
    legend: Dict[str, Any], task_name: str, allowed_metric_keys: List[str]
) -> Dict[str, str]:
    """Looks up legend["Task-Specific Metrics"][task_name] by normalized-label
    match, filtered to only the metric names this task actually reports
    (e.g. excludes GenPPL for tasks the callback doesn't compute it for)."""
    by_task = legend.get("Task-Specific Metrics", {})
    target = _normalize_label(task_name)

    for label, entries in by_task.items():
        if _normalize_label(label) == target:
            allowed = set(allowed_metric_keys)
            return {k: v for k, v in entries.items() if k in allowed}

    return {}


def _render_fraction(numerator: str, denominator: str) -> str:
    return (
        f'<span class="frac">'
        f'<span class="num">{_esc(numerator)}</span>'
        f'<span class="den">{_esc(denominator)}</span>'
        f"</span>"
    )


def _render_explanation(text: str) -> str:
    """Escapes explanation text, rendering any {num}/{den} markers as a
    stacked fraction instead of literal braces/slash."""
    parts = []
    last = 0
    for m in _FRACTION_RE.finditer(text):
        parts.append(_esc(text[last:m.start()]))
        parts.append(_render_fraction(m.group(1).strip(), m.group(2).strip()))
        last = m.end()
    parts.append(_esc(text[last:]))
    return "".join(parts)


def _render_legend(legend: Dict[str, Dict[str, str]]) -> str:
    fields = legend.get("Fields", {})
    metrics = {
        name: explanation
        for name, explanation in legend.get("Metrics", {}).items()
        if _legend_metric_abbrev(name) in LEGEND_METRICS_SHOWN
    }

    if not fields and not metrics:
        return ""

    def _group(title: str, entries: Dict[str, str], *, with_arrows: bool) -> str:
        if not entries:
            return ""
        items = ""
        for name, explanation in entries.items():
            arrow = _direction_arrow(_legend_metric_abbrev(name)) if with_arrows else ""
            items += f"<dt>{_esc(name)}{_esc(arrow)}</dt><dd>{_render_explanation(explanation)}</dd>"
        return (
            f'<div class="legend-group">'
            f"<h3>{_esc(title)}</h3>"
            f'<dl class="legend-list">{items}</dl>'
            f"</div>"
        )

    groups = (
        _group("Fields", fields, with_arrows=False)
        + _group("Metrics", metrics, with_arrows=True)
    )

    return (
        f'<section id="legend">\n'
        f"<h2>Legend</h2>\n"
        f'<div class="legend-groups">{groups}</div>\n'
        f"</section>\n"
    )


def _render_samplers_table(samplers_list: list) -> str:
    """Standalone table listing every resolved sampler spec (tag, algorithm,
    steps/NFE, churn) -- replaces cramming this into the header meta grid."""
    if not samplers_list:
        return ""

    headers = ["Tag", "Sampler", "Steps", "Target NFE", "Actual NFE", "Stochastic", "S-Churn", "σ_min"]
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    rows = ""
    for s in samplers_list:
        stochastic = bool(s.get("stochastic_enabled", False))
        s_churn = s.get("s_churn")
        cells = "".join([
            f'<td class="sampler-col">{_esc(s.get("name", "?"))}</td>',
            f'<td>{_esc(s.get("sampler_name", "?"))}</td>',
            f'<td class="num-col">{_esc(s.get("num_steps", "?"))}</td>',
            f'<td class="num-col">{_esc(s.get("target_nfe", "?"))}</td>',
            f'<td class="num-col">{_esc(s.get("actual_nfe", "?"))}</td>',
            f'<td>{"Yes" if stochastic else "No"}</td>',
            f'<td class="num-col">{_esc(s_churn) if s_churn is not None else "—"}</td>',
            f'<td class="num-col">{_esc(s.get("terminal_sigma", "?"))}</td>',
        ])
        rows += f"<tr>{cells}</tr>\n"

    return (
        f'<section id="samplers">\n'
        f"<h2>Sampler Configurations</h2>\n"
        f'<div class="metrics-compare-wrap">\n'
        f'<table class="metrics-compare-table">\n'
        f"<thead><tr>{th}</tr></thead>\n"
        f"<tbody>\n{rows}</tbody>\n"
        f"</table>\n</div>\n"
        f"</section>\n"
    )


def _is_valid_metric_value(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _render_metrics_comparison(task_key: str, sampler_results: Dict[str, dict]) -> str:
    """Compact table comparing metrics across all samplers for one task.
    The best value in each metric column (per METRIC_DIRECTION) is bolded."""
    metrics_meta = TASK_META.get(task_key, {}).get("metrics", {})
    metric_keys = list(metrics_meta.keys())
    if not metric_keys:
        return ""

    th = "<th>Sampler</th>" + "".join(
        f"<th>{_esc(metrics_meta[mk]['name'])}{_esc(_direction_arrow(mk))}</th>" for mk in metric_keys
    )

    # Best value per metric column; columns with no known direction are left unmarked.
    best_per_metric: Dict[str, float] = {}
    for mk in metric_keys:
        higher_is_better = METRIC_DIRECTION.get(_infer_metric_type(mk))
        if higher_is_better is None:
            continue
        values = [
            v for v in (task_data.get("metrics", {}).get(mk) for task_data in sampler_results.values())
            if _is_valid_metric_value(v)
        ]
        if values:
            best_per_metric[mk] = max(values) if higher_is_better else min(values)

    rows = ""
    for sampler_name, task_data in sampler_results.items():
        metrics = task_data.get("metrics", {})
        cells = f'<td class="sampler-col">{_esc(sampler_name)}</td>'
        for mk in metric_keys:
            val = metrics.get(mk)
            is_best = (
                mk in best_per_metric
                and _is_valid_metric_value(val)
                and math.isclose(val, best_per_metric[mk], rel_tol=1e-9, abs_tol=1e-12)
            )
            cls = "metric-col best" if is_best else "metric-col"
            cells += f'<td class="{cls}">{_fmt_metric(val)}</td>'
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


def _render_task_specific_metrics(entries: Dict[str, str]) -> str:
    """Renders the "this task's metrics mean X" block shown right under the
    task description, sourced from legend.txt's "Task-Specific Metrics:"
    section (empty for tasks with nothing task-specific to say, e.g. STT/
    Continuation whose WER/CER keep their generic meaning)."""
    if not entries:
        return ""
    items = ""
    for name, explanation in entries.items():
        arrow = _direction_arrow(name)
        items += f"<dt>{_esc(name)}{_esc(arrow)}</dt><dd>{_render_explanation(explanation)}</dd>"
    return f'<dl class="legend-list task-specific-metrics">{items}</dl>\n'


def _render_task_section(
    task_key: str,
    sampler_results: Dict[str, dict],
    task_idx: int,
    task_specific_entries: Dict[str, str] | None = None,
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
    task_specific_html = _render_task_specific_metrics(task_specific_entries or {})

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
        f'{task_specific_html}'
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


def _render_html(data: dict, legend: Dict[str, Any] | None = None) -> str:
    data = _normalize_data(data)
    header = data.get("header", {})
    samplers_data = data.get("samplers", {})  # {sampler_name: {tasks: {task: data}}}

    # Reorganise to {task: {sampler_name: task_data}} for rendering
    task_sampler_results: Dict[str, Dict[str, dict]] = {}
    for sampler_name, sd in samplers_data.items():
        for task_key, task_data in sd.get("tasks", {}).items():
            task_sampler_results.setdefault(task_key, {})[sampler_name] = task_data

    samplers_list = header.get("samplers", [])
    legend_html = _render_legend(legend or {})

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
    ])

    samplers_table_html = _render_samplers_table(samplers_list)

    # ── Nav links ───────────────────────────────────────────
    nav_links = "".join(
        [f'<a href="#legend">Legend</a>\n'] if legend_html else []
    ) + "".join(
        [f'<a href="#samplers">Samplers</a>\n'] if samplers_list else []
    ) + "".join(
        f'<a href="#{tk}">{_esc(TASK_META.get(tk, {}).get("name", tk))}</a>\n'
        for tk in TASK_ORDER
        if tk in task_sampler_results
    )

    # ── Task sections ───────────────────────────────────────
    sections = ""
    idx = 0
    for tk in TASK_ORDER:
        if tk in task_sampler_results:
            task_specific_entries = _task_specific_metrics_for(
                legend or {}, TASK_META.get(tk, {}).get("name", tk),
                list(TASK_META.get(tk, {}).get("metrics", {}).keys()),
            )
            sections += _render_task_section(tk, task_sampler_results[tk], idx, task_specific_entries)
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
  {legend_html}
  {samplers_table_html}
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

    legend = _parse_legend(LEGEND_PATH)
    html_str = _render_html(data, legend=legend)
    out_path = save_dir / "report.html"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path
