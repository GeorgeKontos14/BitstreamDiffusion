from __future__ import annotations

import html as _html
import json
import math
import re
import shutil
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

# Extra CSS for the combined (multi-sweep) report: the layered NFE -> gamma ->
# guidance selector, and the diagnostics (entropy/loss plot) cards. Kept
# separate from _CSS so the single-sweep report (build_textaudio_report)
# is unaffected.
_LAYERED_CSS = """\
/* ── Layered sampler selector (NFE -> gamma -> guidance) ──── */
.layered-controls {
  display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: center;
  margin-top: 1.5rem; margin-bottom: 1rem;
  padding: 0.85rem 1.1rem; background: var(--bg-alt);
  border: 1px solid var(--border); border-radius: var(--radius);
}
.control-group { display: flex; align-items: center; gap: 0.5rem; }
.control-group label {
  font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted);
}
.control-group select {
  font-family: var(--mono); font-size: 0.85rem; padding: 0.3rem 0.6rem;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg); color: var(--text);
}

/* ── Diagnostics (entropy / loss plots) ──────────────────── */
.diagnostics-grid {
  display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 1rem;
}
.diagnostic-card {
  flex: 1; min-width: 320px; background: var(--bg-alt);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem;
}
.diagnostic-card h3 {
  font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted); margin: 0 0 0.75rem;
}
.diagnostic-card img { width: 100%; height: auto; border-radius: 4px; }
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

# JS for the combined report's layered NFE -> gamma -> guidance selector.
# Each .layered-controls block carries a sibling <script type="application/json">
# with its {nfe: {gamma: {guidance: tag}}} hierarchy; selecting a level
# repopulates the next and shows the matching .sampler-panel (same panel
# markup/attrs as the flat tab-strip in _JS, just driven by three selects
# instead of one row of buttons).
_LAYERED_JS = """\
document.querySelectorAll('.layered-controls').forEach(function(ctrl) {
  var task = ctrl.dataset.task;
  var dataEl = document.querySelector('.hierarchy-data[data-task="' + task + '"]');
  var hierarchy = JSON.parse(dataEl.textContent);
  var nfeSel = ctrl.querySelector('.sel-nfe');
  var gammaSel = ctrl.querySelector('.sel-gamma');
  var gsSel = ctrl.querySelector('.sel-gs');
  var gsGroup = ctrl.querySelector('.sel-gs-group');

  function optionLabel(key) {
    if (key === 'deterministic') return 'Deterministic';
    if (key === '0') return 'None';
    return key;
  }

  function populate(sel, keys) {
    sel.innerHTML = '';
    keys.forEach(function(k) {
      var opt = document.createElement('option');
      opt.value = k;
      opt.textContent = optionLabel(k);
      sel.appendChild(opt);
    });
  }

  function sortNumericFirst(keys, firstKey) {
    return keys.slice().sort(function(a, b) {
      if (a === firstKey) return -1;
      if (b === firstKey) return 1;
      return parseFloat(a) - parseFloat(b);
    });
  }

  function showPanel(tag) {
    document.querySelectorAll('.sampler-panel[data-task="' + task + '"]').forEach(function(p) {
      p.style.display = 'none';
    });
    if (!tag) return;
    var active = document.querySelector(
      '.sampler-panel[data-task="' + task + '"][data-sampler="' + tag + '"]'
    );
    if (active) active.style.display = 'block';
  }

  function onGsChange() {
    var nfe = nfeSel.value, gamma = gammaSel.value, gs = gsSel.value;
    var tag = ((hierarchy[nfe] || {})[gamma] || {})[gs];
    showPanel(tag);
  }

  function onGammaChange() {
    var nfe = nfeSel.value, gamma = gammaSel.value;
    var gsKeys = Object.keys((hierarchy[nfe] || {})[gamma] || {});
    gsGroup.style.display = gsKeys.length > 1 ? '' : 'none';
    populate(gsSel, sortNumericFirst(gsKeys, '0'));
    onGsChange();
  }

  function onNfeChange() {
    var nfe = nfeSel.value;
    var gammaKeys = Object.keys(hierarchy[nfe] || {});
    populate(gammaSel, sortNumericFirst(gammaKeys, 'deterministic'));
    onGammaChange();
  }

  nfeSel.addEventListener('change', onNfeChange);
  gammaSel.addEventListener('change', onGammaChange);
  gsSel.addEventListener('change', onGsChange);

  populate(nfeSel, sortNumericFirst(Object.keys(hierarchy), '__none__'));
  onNfeChange();
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


def _render_legend(legend: Dict[str, Dict[str, str]], allowed_metrics: set | None = None) -> str:
    allowed = allowed_metrics if allowed_metrics is not None else LEGEND_METRICS_SHOWN
    fields = legend.get("Fields", {})
    metrics = {
        name: explanation
        for name, explanation in legend.get("Metrics", {}).items()
        if _legend_metric_abbrev(name) in allowed
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

    headers = ["Tag", "Sampler", "NFE", "Stochastic", "Gamma", "Guidance"]
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    rows = ""
    for s in samplers_list:
        stochastic = bool(s.get("stochastic_enabled", False))
        gamma = s.get("gamma")
        gamma_str = f"{gamma:.4g}" if isinstance(gamma, (int, float)) else "—"
        guidance = s.get("guidance_scale")
        guidance_str = f"{guidance:g}" if isinstance(guidance, (int, float)) and guidance > 0 else "—"
        cells = "".join([
            f'<td class="sampler-col">{_esc(s.get("name", "?"))}</td>',
            f'<td>{_esc(s.get("sampler_name", "?"))}</td>',
            f'<td class="num-col">{_esc(s.get("actual_nfe", "?"))}</td>',
            f'<td>{"Yes" if stochastic else "No"}</td>',
            f'<td class="num-col">{_esc(gamma_str)}</td>',
            f'<td class="num-col">{_esc(guidance_str)}</td>',
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


# ── Layered NFE -> gamma -> guidance selector (combined multi-sweep report) ──

# Sampler tags are built by _build_sampler_specs (utils/callbacks/
# textaudio_generation.py): "{sampler_name}_nfe{N}[_stoch-g{gamma}][_gs{scale}]".
# e.g. "ddim_entropic_nfe256", "ddim_entropic_nfe256_stoch-g0.17",
# "ddim_entropic_nfe256_stoch-g0.17_gs2".
_SAMPLER_TAG_RE = re.compile(
    r'^(?P<base>.+?)_nfe(?P<nfe>\d+)(?:_stoch-g(?P<gamma>[0-9.]+))?(?:_gs(?P<gs>[0-9.]+))?$'
)


def _parse_sampler_tag(tag: str) -> Dict[str, str]:
    """Splits a sampler tag into its NFE / gamma / guidance-scale axes for the
    layered selector. gamma="deterministic" and gs="0" stand in for "not
    present in the tag" (no stochastic churn / no guidance), matching how
    _build_sampler_specs only appends those suffixes when applicable."""
    m = _SAMPLER_TAG_RE.match(tag)
    if not m:
        # Unrecognized shape -- still reachable in the report, just on its
        # own branch, rather than silently dropped from the combined view.
        return {"base": tag, "nfe": "?", "gamma": "deterministic", "gs": "0"}
    return {
        "base": m.group("base"),
        "nfe": m.group("nfe"),
        "gamma": m.group("gamma") or "deterministic",
        "gs": m.group("gs") or "0",
    }


def _build_hierarchy(sampler_results: Dict[str, dict]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{nfe: {gamma_or_"deterministic": {guidance_or_"0": tag}}} for every
    sampler tag present in this task's results, used to drive the client-side
    layered selector (see _LAYERED_JS)."""
    hierarchy: Dict[str, Dict[str, Dict[str, str]]] = {}
    for tag in sampler_results:
        parsed = _parse_sampler_tag(tag)
        nfe_bucket = hierarchy.setdefault(parsed["nfe"], {})
        gamma_bucket = nfe_bucket.setdefault(parsed["gamma"], {})
        gamma_bucket[parsed["gs"]] = tag
    return hierarchy


def _render_layered_controls(task_key: str, hierarchy: dict) -> str:
    # "</" can't appear in a JSON string here (tags/keys are our own
    # alphanumeric/dot strings), but escape defensively since this is
    # embedded verbatim inside a <script> block.
    hierarchy_json = json.dumps(hierarchy).replace("</", "<\\/")
    return (
        f'<div class="layered-controls" data-task="{_esc(task_key)}">\n'
        f'  <div class="control-group"><label>NFE</label><select class="sel-nfe"></select></div>\n'
        f'  <div class="control-group"><label>Gamma</label><select class="sel-gamma"></select></div>\n'
        f'  <div class="control-group sel-gs-group"><label>Guidance</label><select class="sel-gs"></select></div>\n'
        f'</div>\n'
        f'<script type="application/json" class="hierarchy-data" data-task="{_esc(task_key)}">'
        f'{hierarchy_json}</script>\n'
    )


def _render_task_section_combined(
    task_key: str,
    sampler_results: Dict[str, dict],
    task_idx: int,
    task_specific_entries: Dict[str, str] | None = None,
) -> str:
    """Same as _render_task_section, except sample panels are picked via the
    layered NFE -> gamma -> guidance selector instead of a flat tab strip."""
    meta = TASK_META.get(task_key, {})
    name = meta.get("name", task_key)
    desc = meta.get("description", "")
    columns = meta.get("columns", [])
    metrics_meta = meta.get("metrics", {})
    task_specific_html = _render_task_specific_metrics(task_specific_entries or {})

    sampler_names = list(sampler_results.keys())
    multi = len(sampler_names) > 1

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

    controls_html = _render_layered_controls(task_key, _build_hierarchy(sampler_results)) if sampler_names else ""

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
        f'{controls_html}'
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


def _render_html_combined(
    data: dict,
    legend: Dict[str, Any] | None = None,
    *,
    sigma_data: float | None = None,
    entropy_plot_rel: str | None = None,
    loss_plot_rel: str | None = None,
    sweep_names: List[str] | None = None,
) -> str:
    """Same overall structure as _render_html, for a *merged* multi-sweep
    dataset (see build_combined_textaudio_report): sample panels are picked
    via the layered NFE -> gamma -> guidance selector instead of a flat tab
    strip, an optional σ_data meta item is added, and an optional
    "Training Diagnostics" section (entropy/loss plots) is inserted right
    above the legend -- entirely omitted when neither plot is supplied."""
    data = _normalize_data(data)
    header = data.get("header", {})
    samplers_data = data.get("samplers", {})

    task_sampler_results: Dict[str, Dict[str, dict]] = {}
    for sampler_name, sd in samplers_data.items():
        for task_key, task_data in sd.get("tasks", {}).items():
            task_sampler_results.setdefault(task_key, {})[sampler_name] = task_data

    samplers_list = header.get("samplers", [])
    legend_html = _render_legend(legend or {})

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

    step = header.get("step", "?")
    meta_parts = [
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
    ]
    if sigma_data is not None:
        meta_parts.append(_render_meta_item("Estimated σ_data", f"{sigma_data:.4f}"))
    meta_items = "".join(meta_parts)

    samplers_table_html = _render_samplers_table(samplers_list)

    # ── Diagnostics: entropy / loss plots, only if supplied ──
    diag_cards = []
    if entropy_plot_rel:
        diag_cards.append(
            f'<div class="diagnostic-card"><h3>Entropy Schedule</h3>'
            f'<img src="{_esc(entropy_plot_rel)}" alt="Entropy schedule plot"></div>'
        )
    if loss_plot_rel:
        diag_cards.append(
            f'<div class="diagnostic-card"><h3>Training Loss</h3>'
            f'<img src="{_esc(loss_plot_rel)}" alt="Training loss plot"></div>'
        )
    diagnostics_html = (
        f'<section id="diagnostics">\n<h2>Training Diagnostics</h2>\n'
        f'<div class="diagnostics-grid">{"".join(diag_cards)}</div>\n</section>\n'
    ) if diag_cards else ""

    nav_links = "".join(
        [f'<a href="#diagnostics">Diagnostics</a>\n'] if diagnostics_html else []
    ) + "".join(
        [f'<a href="#legend">Legend</a>\n'] if legend_html else []
    ) + "".join(
        [f'<a href="#samplers">Samplers</a>\n'] if samplers_list else []
    ) + "".join(
        f'<a href="#{tk}">{_esc(TASK_META.get(tk, {}).get("name", tk))}</a>\n'
        for tk in TASK_ORDER
        if tk in task_sampler_results
    )

    sections = ""
    idx = 0
    for tk in TASK_ORDER:
        if tk in task_sampler_results:
            task_specific_entries = _task_specific_metrics_for(
                legend or {}, TASK_META.get(tk, {}).get("name", tk),
                list(TASK_META.get(tk, {}).get("metrics", {}).keys()),
            )
            sections += _render_task_section_combined(tk, task_sampler_results[tk], idx, task_specific_entries)
            idx += 1

    title = _esc(header.get("title", "TextAudio Evaluation")) + " (Combined)"
    footer_suffix = f" -- sweeps: {', '.join(sorted(sweep_names))}" if sweep_names else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Step {_esc(str(step))}</title>
<style>
{_CSS}
{_LAYERED_CSS}
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
  {diagnostics_html}
  {legend_html}
  {samplers_table_html}
  {sections}
  <div class="footer">
    Generated by generate_report.py{_esc(footer_suffix)}
  </div>
</div>
<script>
{_LAYERED_JS}
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


def _resolve_diagnostic_asset(src) -> Path:
    """*src* may omit its extension (as typed by a person); try common image
    extensions before giving up."""
    src_path = Path(src)
    if src_path.exists():
        return src_path
    for ext in (".png", ".svg", ".jpg", ".jpeg"):
        candidate = Path(str(src_path) + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Diagnostic plot not found: {src}")


def _copy_diagnostic_asset(src, step_dir: Path, base_name: str) -> str | None:
    """Copies *src* into <step_dir>/assets/<base_name><ext> so the combined
    report stays self-contained (portable if step_dir is copied elsewhere),
    same as how sample .wav files already live under step_dir. Returns the
    path relative to step_dir for use as an <img src>, or None if src is
    falsy (the plot was omitted -- the caller skips it entirely)."""
    if not src:
        return None
    src_path = _resolve_diagnostic_asset(src)
    assets_dir = step_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / f"{base_name}{src_path.suffix}"
    shutil.copy2(src_path, dest)
    return f"assets/{dest.name}"


def build_combined_textaudio_report(
    step_dir,
    *,
    sigma_data: float | None = None,
    entropy_plot=None,
    loss_plot=None,
    output_path=None,
) -> Path:
    """
    Merge every evaluation sweep under *step_dir* (each an immediate
    subdirectory with its own data.json, e.g. nfe256_sweep25/, nfe512_sweep25/
    -- see scripts/multimodal_textaudio/offline_generation.py) into one
    combined report.html, written directly into step_dir.

    Parameters
    ----------
    step_dir : str or Path
        e.g. runs/COBIT_630M_632/textaudio_offline/step_000372000 -- the
        directory containing one subfolder per sweep.
    sigma_data : float, optional
        Estimated sigma_data to display in the header. Omitted entirely from
        the report if not given.
    entropy_plot, loss_plot : str or Path, optional
        Paths to the latest entropy-schedule / loss plots. Each is copied
        into step_dir/assets/ and shown in a "Training Diagnostics" section
        right above the legend. That section (and each card within it) is
        omitted entirely when its plot isn't supplied.
    output_path : str or Path, optional
        Defaults to step_dir/combined_report.html.

    Returns
    -------
    Path to the generated report.html.
    """
    step_dir = Path(step_dir)
    sweep_dirs = sorted(p.parent for p in step_dir.glob("*/data.json"))
    if not sweep_dirs:
        raise FileNotFoundError(f"No sweep data.json files found under {step_dir}")

    merged_header: Dict[str, Any] | None = None
    merged_samplers: Dict[str, Any] = {}
    all_sampler_specs: List[dict] = []
    sweep_names: List[str] = []

    for sweep_dir in sweep_dirs:
        with open(sweep_dir / "data.json", "r", encoding="utf-8") as f:
            data = _normalize_data(json.load(f))
        sweep_name = sweep_dir.name
        sweep_names.append(sweep_name)

        header = data.get("header", {})
        if merged_header is None:
            merged_header = dict(header)
        all_sampler_specs.extend(header.get("samplers", []))

        for sampler_tag, sd in data.get("samplers", {}).items():
            if sampler_tag in merged_samplers:
                raise ValueError(
                    f"Sampler tag '{sampler_tag}' appears in more than one sweep under "
                    f"{step_dir} (also in {sweep_name}) -- cannot merge unambiguously."
                )
            for task_data in sd.get("tasks", {}).values():
                # Sample media paths are relative to their *own* sweep_dir;
                # re-root them under sweep_name since the combined report
                # lives directly in step_dir (their parent).
                for sample in task_data.get("samples", []):
                    if sample.get("gen_wav"):
                        sample["gen_wav"] = f"{sweep_name}/{sample['gen_wav']}"
                    if sample.get("ref_wav"):
                        sample["ref_wav"] = f"{sweep_name}/{sample['ref_wav']}"
            merged_samplers[sampler_tag] = sd

    assert merged_header is not None
    merged_header["samplers"] = all_sampler_specs
    merged_data = {"header": merged_header, "samplers": merged_samplers}

    legend = _parse_legend(LEGEND_PATH)
    entropy_plot_rel = _copy_diagnostic_asset(entropy_plot, step_dir, "entropy_plot")
    loss_plot_rel = _copy_diagnostic_asset(loss_plot, step_dir, "loss_plot")

    html_str = _render_html_combined(
        merged_data,
        legend=legend,
        sigma_data=sigma_data,
        entropy_plot_rel=entropy_plot_rel,
        loss_plot_rel=loss_plot_rel,
        sweep_names=sweep_names,
    )

    out_path = Path(output_path) if output_path else step_dir / "combined_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluation protocol report
#
# A separate rendering path from everything above (which serves the
# training-time visualization callback's data.json). This one reads directly
# from a completed evaluation run_dir -- run_dir/{tag}/manifest.json +
# run_dir/{tag}/results.json + run_dir/{tag}/{task}/samples.json, written
# incrementally by evaluation/evaluation_drivers/textaudio_{generate,eval}.py
# -- and adds literature-baseline comparison tables. Shares CSS, cell
# rendering, legend parsing, and direction-arrow inference with the code
# above; TASK_META/TASK_ORDER/_render_html* above are untouched.
# ═══════════════════════════════════════════════════════════════════════════

import csv

EVAL_TASK_META: Dict[str, Dict[str, Any]] = {
    "joint": {
        "name": "Joint Generation",
        "description": TASK_META["joint"]["description"],
        "metrics": {
            "Cross-Modal WER": {"name": "Cross-Modal WER"},
            "Cross-Modal CER": {"name": "Cross-Modal CER"},
            "GenPPL-text": {"name": "GenPPL-text"},
            "GenPPL-speech": {"name": "GenPPL-speech"},
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
        "description": TASK_META["tts"]["description"],
        "metrics": {
            "ASR-WER": {"name": "ASR-WER"},
            "ASR-CER": {"name": "ASR-CER"},
            "UTMOS": {"name": "UTMOS"},
            "SpkSim": {"name": "SpkSim"},
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
        "description": TASK_META["stt"]["description"],
        "metrics": {
            "clean-WER": {"name": "Clean-WER"},
            "other-WER": {"name": "Other-WER"},
        },
        "columns": [
            ("partition", "Partition", "text"),
            ("ref_wav", "Reference Audio", "audio"),
            ("ref_text", "Reference Text", "text"),
            ("gen_text", "Generated Text", "text"),
        ],
    },
    "cont": {
        "name": "Continuation",
        "description": (
            TASK_META["cont"]["description"]
            + " GenPPL-speech and FSD are taken from the cont_flow_slm task's "
              "results for the same sampler tag (its own dedicated cache), "
              "not from this task's own results.json."
        ),
        "metrics": {
            "LLM-as-a-Judge": {"name": "LLM-as-Judge"},
            "UTMOS": {"name": "UTMOS"},
            "GenPPL-speech": {"name": "GenPPL-speech"},
            "FSD": {"name": "FSD"},
        },
        "columns": [
            ("gen_text", "Generated Text", "text"),
            ("gen_wav", "Generated Audio", "audio"),
        ],
    },
    "salmon": {
        "name": "SALMON",
        "description": (
            "Assesses recognition of paralinguistic properties (sentiment, speaker, "
            "gender, background domain/randomness, room acoustics) via a generative "
            "continuation judged against the true and a synthetic alternative."
        ),
        "metrics": {
            "sentiment_consistency": {"name": "Sentiment"},
            "speaker_consistency": {"name": "Speaker"},
            "gender_consistency": {"name": "Gender"},
            "bg_domain_consistency": {"name": "Bg (Domain)"},
            "bg_all_consistency": {"name": "Bg (Random)"},
            "rir_consistency": {"name": "Room"},
        },
        # No "columns" key: metrics-only section, no sample table (per spec --
        # SALMON's own samples.json rows carry no reference/text worth showing).
    },
}
EVAL_TASK_ORDER = ["joint", "tts", "stt", "cont", "salmon"]

# Our SALMON metric keys ("sentiment_consistency" etc.) don't contain the
# literal substring "SALMON", so the shared pattern list needs this extra
# entry to classify them for direction-arrow/percent-formatting purposes.
# Safe to add: no pre-existing metric key anywhere contains "CONSISTENCY".
if not any(p == "CONSISTENCY" for p, _ in _METRIC_TYPE_PATTERNS):
    _METRIC_TYPE_PATTERNS.insert(0, ("CONSISTENCY", "salmon"))

# Metric types shown as a percentage. Our own results.json values are 0-1
# fractions; baseline CSVs already store these as 0-100 percentages (per
# assets/textaudio_baselines/*.csv convention) -- see already_percent below.
_PERCENT_METRIC_TYPES = {"wer", "cer", "salmon"}


def _fmt_metric_pct(value: Any, metric_key: str, *, already_percent: bool) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if not isinstance(value, (int, float)):
        return _esc(value)
    mtype = _infer_metric_type(metric_key)
    if mtype in _PERCENT_METRIC_TYPES:
        pct = float(value) if already_percent else float(value) * 100.0
        return f"{pct:.2f}%"
    return f"{float(value):.4f}"


def _load_checkpoint_meta(checkpoint_path) -> tuple[int | None, int | None]:
    """(total_param_count, global_step) read straight from a training
    checkpoint's state dict -- CPU-only, no model instantiation needed."""
    import torch
    path = Path(checkpoint_path)
    if not path.exists():
        return None, None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    step = ckpt.get("global_step") if isinstance(ckpt, dict) else None
    return n_params, step


def _fmt_params_millions(n: int) -> str:
    return f"{n / 1e6:.1f}M"


# ── Baselines (assets/textaudio_baselines/{task}.csv) ───────────────────────

BASELINE_DIR = Path(__file__).resolve().parent.parent / "assets" / "textaudio_baselines"
_BASELINE_FIXED_COLS = ("Model", "Parameters", "Type", "URL")

# Our metric key -> baseline CSV column name, for the (task, metric) pairs
# both sides report but under different names/splits (we split GenPPL into
# -text/-speech, baselines report one GenPPL; stt's casing differs). Metrics
# only one side reports (e.g. our stt CER, a baseline's LLM-as-Judge) don't
# need an entry here -- _eval_table_columns adds them as their own column via
# a union, not an intersection, so they're never silently dropped.
BASELINE_METRIC_MAP: Dict[str, Dict[str, str]] = {
    "tts": {"ASR-WER": "ASR-WER", "ASR-CER": "ASR-CER", "SpkSim": "SpkSim", "UTMOS": "UTMOS"},
    "stt": {"clean-WER": "Clean-WER", "other-WER": "Other-WER"},
    "cont": {
        "GenPPL-speech": "GenPPL", "FSD": "FSD", "UTMOS": "UTMOS",
        "LLM-as-a-Judge": "LLM-as-Judge",
    },
    "salmon": {
        "sentiment_consistency": "Sentiment", "speaker_consistency": "Speaker",
        "gender_consistency": "Gender", "bg_domain_consistency": "Bg (domain)",
        "bg_all_consistency": "Bg (random)", "rir_consistency": "Room",
    },
}


def load_baselines(task_key: str) -> List[Dict[str, str]]:
    path = BASELINE_DIR / f"{task_key}.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def _baseline_metric_columns(task_key: str) -> List[str]:
    """Ordered baseline CSV columns that aren't Model/Parameters/Type/URL."""
    path = BASELINE_DIR / f"{task_key}.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    return [c for c in fieldnames if c not in _BASELINE_FIXED_COLS]


def _eval_table_columns(task_key: str) -> List[tuple]:
    """(display_key, our_metric_key_or_None, baseline_col_or_None) for every
    metric either we or any baseline reports for this task -- a union, so a
    metric only one side has still gets its own column (blank on the other
    side) instead of being dropped."""
    our_to_baseline = BASELINE_METRIC_MAP.get(task_key, {})
    our_keys = list(EVAL_TASK_META.get(task_key, {}).get("metrics", {}).keys())
    columns = [(k, k, our_to_baseline.get(k)) for k in our_keys]
    covered_baseline_cols = set(our_to_baseline.values())
    for bc in _baseline_metric_columns(task_key):
        if bc not in covered_baseline_cols:
            columns.append((bc, None, bc))
    return columns


def _render_eval_metrics_table(
    task_key: str,
    sampler_results: Dict[str, dict],
    baselines: List[Dict[str, str]],
    checkpoint_params: int | None,
) -> str:
    """Baseline rows (Model linked to its paper, Parameters, Type, then every
    metric column) followed by our own sampler row(s) (Parameters from the
    checkpoint, no Type/URL). Used for every task, including joint/salmon --
    joint just has an empty baselines list (no assets/.../joint.csv)."""
    columns = _eval_table_columns(task_key)
    if not columns:
        return '<p class="no-data">No metrics defined for this task.</p>'

    metrics_meta = EVAL_TASK_META.get(task_key, {}).get("metrics", {})
    header_labels = [
        metrics_meta[ok]["name"] if ok else bk for _, ok, bk in columns
    ]
    th = "<th>Model</th><th>Parameters</th><th>Type</th>" + "".join(
        f"<th>{_esc(label)}{_esc(_direction_arrow(ok or bk))}</th>"
        for label, (_, ok, bk) in zip(header_labels, columns)
    )

    rows = ""
    for b in baselines:
        model = _esc(b.get("Model", "?"))
        url = (b.get("URL") or "").strip()
        model_cell = (
            f'<a href="{_esc(url)}" target="_blank" rel="noopener">{model}</a>' if url else model
        )
        cells = [
            f'<td class="sampler-col">{model_cell}</td>',
            f'<td class="num-col">{_esc(b.get("Parameters") or "—")}</td>',
            f'<td>{_esc(b.get("Type") or "—")}</td>',
        ]
        for _, ok, bk in columns:
            raw = (b.get(bk) or "").strip() if bk else ""
            if not raw or raw == "-":
                cells.append('<td class="metric-col">—</td>')
            else:
                try:
                    cells.append(
                        f'<td class="metric-col">{_fmt_metric_pct(float(raw), bk, already_percent=True)}</td>'
                    )
                except ValueError:
                    cells.append(f'<td class="metric-col">{_esc(raw)}</td>')
        rows += f"<tr>{''.join(cells)}</tr>\n"

    params_str = _fmt_params_millions(checkpoint_params) if checkpoint_params else "—"
    for tag, task_data in sampler_results.items():
        metrics = task_data.get("metrics", {})
        cells = [
            f'<td class="sampler-col">{_esc(tag)}</td>',
            f'<td class="num-col">{_esc(params_str)}</td>',
            '<td>—</td>',
        ]
        for _, ok, bk in columns:
            val = metrics.get(ok) if ok else None
            cells.append(
                f'<td class="metric-col">{_fmt_metric_pct(val, ok or bk, already_percent=False)}</td>'
            )
        rows += f"<tr>{''.join(cells)}</tr>\n"

    return (
        f'<div class="metrics-compare-wrap">\n<table class="metrics-compare-table">\n'
        f"<thead><tr>{th}</tr></thead>\n<tbody>\n{rows}</tbody>\n</table>\n</div>"
    )


def _render_eval_task_section(
    task_key: str,
    sampler_results: Dict[str, dict],
    baselines: List[Dict[str, str]],
    checkpoint_params: int | None,
    task_idx: int,
) -> str:
    meta = EVAL_TASK_META.get(task_key, {})
    name = meta.get("name", task_key)
    desc = meta.get("description", "")
    columns = meta.get("columns")  # None for salmon -> metrics-only section

    metrics_html = _render_eval_metrics_table(task_key, sampler_results, baselines, checkpoint_params)

    if columns is None:
        return (
            f'<section id="eval-{task_key}">\n'
            f"<h2>{task_idx + 1}. {_esc(name)}</h2>\n"
            f'<p class="task-desc">{_esc(desc)}</p>\n'
            f'{metrics_html}\n'
            f"</section>\n"
        )

    sampler_names = list(sampler_results.keys())
    multi = len(sampler_names) > 1

    tab_strip = ""
    if multi:
        tabs = "".join(
            f'<button class="tab-btn{" active" if i == 0 else ""}" '
            f'data-task="eval-{_esc(task_key)}" data-sampler="{_esc(sname)}">'
            f'{_esc(sname)}</button>\n'
            for i, sname in enumerate(sampler_names)
        )
        tab_strip = f'<div class="tab-strip">{tabs}</div>\n'

    panels = ""
    for i, (sname, task_data) in enumerate(sampler_results.items()):
        style = '' if i == 0 else ' style="display:none"'
        data_attrs = f'data-task="eval-{_esc(task_key)}" data-sampler="{_esc(sname)}"'
        samples = task_data.get("samples", [])
        total = task_data.get("total_samples", len(samples))
        caption = (
            f'<p class="task-desc">Showing {len(samples)} of {total:,} samples.</p>\n'
            if total > len(samples) else ""
        )
        table_html = _render_sample_table(columns, samples)
        panels += (
            f'<div class="sampler-panel" {data_attrs}{style}>\n'
            f'{caption}{table_html}\n'
            f'</div>\n'
        )

    return (
        f'<section id="eval-{task_key}">\n'
        f"<h2>{task_idx + 1}. {_esc(name)}</h2>\n"
        f'<p class="task-desc">{_esc(desc)}</p>\n'
        f'{metrics_html}\n'
        f'{tab_strip}'
        f'{panels}'
        f"</section>\n"
    )


# ── Discovery + sample loading from a completed eval run_dir ────────────────

def _discover_eval_tag_manifests(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """{tag: manifest_header} for every run_dir/{tag}/manifest.json -- mirrors
    evaluation/evaluation_drivers/textaudio_eval.py::_discover_tag_manifests
    (kept as a small local copy rather than a cross-package import)."""
    out: Dict[str, Dict[str, Any]] = {}
    for manifest_path in sorted(run_dir.glob("*/manifest.json")):
        tag = manifest_path.parent.name
        try:
            out[tag] = json.loads(manifest_path.read_text(encoding="utf-8"))["header"]
        except Exception:
            continue
    return out


def _load_eval_task_samples(
    run_dir: Path, tag: str, task: str, header: Dict[str, Any], cap: int,
) -> tuple[list, int]:
    """(capped_samples, total_count) for one (tag, task), with gen_wav paths
    rewritten to be run_dir-relative (they're stored task-dir-relative) so
    they resolve correctly from a report.html written directly into run_dir.
    ref_wav is already run_dir-relative (references are shared across tags)."""
    if task == "salmon":
        return [], 0  # metrics-only, no samples shown (per spec)

    if task == "stt":
        partitions = header.get("stt_partitions", ["clean", "other"])
        per_partition_cap = max(1, cap // max(1, len(partitions))) if cap else 0
        samples: list = []
        total = 0
        for p in partitions:
            path = run_dir / tag / f"stt_{p}" / "samples.json"
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            total += len(rows)
            for row in rows[:per_partition_cap]:
                row = dict(row)
                row["partition"] = p
                samples.append(row)
        return samples, total

    path = run_dir / tag / task / "samples.json"
    if not path.exists():
        return [], 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    capped = []
    for row in rows[:cap]:
        row = dict(row)
        if row.get("gen_wav"):
            row["gen_wav"] = f"{tag}/{task}/{row['gen_wav']}"
        capped.append(row)
    return capped, len(rows)


def _render_html_eval(
    run_dir: Path,
    tag_headers: Dict[str, Dict[str, Any]],
    task_sampler_results: Dict[str, Dict[str, dict]],
    baselines_by_task: Dict[str, List[Dict[str, str]]],
    checkpoint_params: int | None,
    global_step: int | None,
    legend: Dict[str, Any] | None = None,
) -> str:
    legend_html = _render_legend(
        legend or {}, allowed_metrics={"WER", "CER", "UTMOS", "GenPPL", "SpkSim", "FSD", "SALMON"},
    )

    any_header = next(iter(tag_headers.values()), {})
    seq = any_header.get("sequence_layout", {})
    text_tok = seq.get("text_tokens", 0) or 0
    speaker_tok = seq.get("speaker_tokens", 0) or 0
    total_tok = seq.get("total_tokens", 0) or 0
    speech_tok = max(0, total_tok - text_tok - speaker_tok)
    layout_str = f"{text_tok} text + {speaker_tok} speaker + {speech_tok} speech = {total_tok} tokens"

    meta_items = "".join([
        _render_meta_item("Experiment", any_header.get("experiment", "unknown")),
        _render_meta_item("Checkpoint", f"Step {global_step}" if global_step is not None else "unknown"),
        _render_meta_item("Text Tokenizer", any_header.get("text_tokenizer", "unknown")),
        _render_meta_item("Speaker Tokenizer", any_header.get("speaker_tokenizer", "unknown")),
        _render_meta_item("Speech Tokenizer", any_header.get("speech_tokenizer", "unknown")),
        _render_meta_item("Bits / Token", any_header.get("bits_per_token", "?")),
        _render_meta_item("Sequence Layout", layout_str),
        _render_meta_item(
            "Parameters",
            _fmt_params_millions(checkpoint_params) if checkpoint_params else "unknown",
        ),
    ])

    samplers_list = [
        {
            "name": tag, "sampler_name": h.get("sampler", {}).get("sampler_name", "?"),
            "actual_nfe": h.get("sampler", {}).get("target_nfe", "?"),
            "stochastic_enabled": h.get("sampler", {}).get("stochastic_enabled", False),
            "gamma": h.get("sampler", {}).get("gamma"),
            "guidance_scale": h.get("sampler", {}).get("guidance_scale"),
        }
        for tag, h in tag_headers.items()
    ]
    samplers_table_html = _render_samplers_table(samplers_list)

    nav_links = "".join(
        [f'<a href="#legend">Legend</a>\n'] if legend_html else []
    ) + "".join(
        [f'<a href="#samplers">Samplers</a>\n'] if samplers_list else []
    ) + "".join(
        f'<a href="#eval-{tk}">{_esc(EVAL_TASK_META.get(tk, {}).get("name", tk))}</a>\n'
        for tk in EVAL_TASK_ORDER
        if tk in task_sampler_results
    )

    sections = ""
    idx = 0
    for tk in EVAL_TASK_ORDER:
        if tk in task_sampler_results:
            sections += _render_eval_task_section(
                tk, task_sampler_results[tk], baselines_by_task.get(tk, []), checkpoint_params, idx,
            )
            idx += 1

    title = f"{_esc(any_header.get('experiment', 'Evaluation'))} — Full Evaluation Protocol"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_CSS}
</style>
</head>
<body>
<nav class="topnav">
  <span class="nav-title">{_esc(any_header.get('experiment', 'Evaluation'))} · Full Evaluation Protocol</span>
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
    Generated by generate_eval_report.py
  </div>
</div>
<script>
{_JS}
</script>
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────

def build_eval_protocol_report(
    run_dir,
    *,
    tags: List[str] | None = None,
    samples_per_task: int = 10,
    checkpoint_path=None,
    output_path=None,
) -> Path:
    """
    Builds the full-evaluation-protocol report for a completed run_dir (e.g.
    runs/textaudio_pilot/evaluation/textaudio_eval/last), reading whatever
    sampler tags/tasks/results already exist on disk (see module docstring
    above for the exact layout). Each sampler tag only reports the tasks its
    own manifest.json lists -- nothing is inferred or assumed complete.

    Parameters
    ----------
    run_dir : str or Path
    tags : list of str, optional
        Restrict to these sampler tags (default: every tag discovered under
        run_dir with a manifest.json).
    samples_per_task : int
        How many samples to display per (tag, task) -- always fewer than the
        real dataset size; metrics are still computed over the full set.
    checkpoint_path : str or Path, optional
        Defaults to the "checkpoint" path recorded in the first tag's
        manifest.json (every tag under one run_dir shares the same checkpoint).
    output_path : str or Path, optional
        Defaults to run_dir/report.html.

    Returns
    -------
    Path to the generated report.html.
    """
    run_dir = Path(run_dir)
    tag_headers = _discover_eval_tag_manifests(run_dir)
    if not tag_headers:
        raise FileNotFoundError(f"No {run_dir}/*/manifest.json found -- run evaluation first.")

    if tags:
        missing = [t for t in tags if t not in tag_headers]
        if missing:
            raise ValueError(f"--tags {missing} not found under {run_dir} (available: {sorted(tag_headers)})")
        tag_headers = {t: tag_headers[t] for t in tags}

    if checkpoint_path is None:
        checkpoint_path = next(iter(tag_headers.values())).get("checkpoint")
    checkpoint_params, global_step = (
        _load_checkpoint_meta(checkpoint_path) if checkpoint_path else (None, None)
    )

    task_sampler_results: Dict[str, Dict[str, dict]] = {}
    for tag, header in tag_headers.items():
        results_path = run_dir / tag / "results.json"
        tag_results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
        for task in header.get("tasks", []):
            if task == "cont_flow_slm":
                # Not its own report section -- GenPPL-speech/FSD are folded
                # into "cont" below (its dedicated cache is more reliable for
                # those two metrics than standard cont's own computation).
                continue
            samples, total = _load_eval_task_samples(run_dir, tag, task, header, samples_per_task)
            metrics = dict(tag_results.get(task, {}))
            if task == "cont":
                flow_slm_metrics = tag_results.get("cont_flow_slm", {})
                for key in ("GenPPL-speech", "FSD"):
                    if key in flow_slm_metrics:
                        metrics[key] = flow_slm_metrics[key]
            task_sampler_results.setdefault(task, {})[tag] = {
                "metrics": metrics,
                "samples": samples,
                "total_samples": total,
            }

    baselines_by_task = {tk: load_baselines(tk) for tk in EVAL_TASK_ORDER}
    legend = _parse_legend(LEGEND_PATH)

    html_str = _render_html_eval(
        run_dir, tag_headers, task_sampler_results, baselines_by_task,
        checkpoint_params, global_step, legend=legend,
    )

    out_path = Path(output_path) if output_path else run_dir / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path


def bundle_eval_protocol_report(
    run_dir,
    dest_dir,
    *,
    tags: List[str] | None = None,
    samples_per_task: int = 10,
    checkpoint_path=None,
) -> Path:
    """
    Re-derives the exact same sample selection build_eval_protocol_report
    would (same tags/cap), then copies ONLY the referenced media (gen_wav +
    ref_wav for the shown samples, never the full task directories) plus a
    freshly-rendered report.html into dest_dir -- a small, self-contained,
    portable copy instead of the full multi-GB run_dir.

    Returns the path to dest_dir/report.html.
    """
    run_dir = Path(run_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tag_headers = _discover_eval_tag_manifests(run_dir)
    if tags:
        tag_headers = {t: tag_headers[t] for t in tags if t in tag_headers}

    copied: set = set()
    for tag, header in tag_headers.items():
        for task in header.get("tasks", []):
            if task == "cont_flow_slm":
                continue  # metrics-only merge into "cont" -- no samples of its own are shown
            samples, _ = _load_eval_task_samples(run_dir, tag, task, header, samples_per_task)
            for row in samples:
                for key in ("gen_wav", "ref_wav"):
                    rel = row.get(key)
                    if not rel or rel in copied:
                        continue
                    src = run_dir / rel
                    if not src.exists():
                        continue
                    dst = dest_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied.add(rel)

    report_path = build_eval_protocol_report(
        run_dir, tags=tags, samples_per_task=samples_per_task,
        checkpoint_path=checkpoint_path, output_path=dest_dir / "report.html",
    )
    return report_path
