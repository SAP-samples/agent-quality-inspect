"""
Generate static HTML detail pages for leaderboard entries.

Improved version with readable folder names:
  {dataset}_{model}_{persona}_{id}/

Example: tau2bench_airline_gpt4_expert_a99aa61b/

Produces:
  demo/agent-eval-dashboard/leaderboard/details/{folder_name}/index.html      – summary + trial list
  demo/agent-eval-dashboard/leaderboard/details/{folder_name}/trial_N.html   – per-trial conversation view

Usage:
    python demo/agent-eval-dashboard/scripts/page_generators/generate_detail_pages_v2.py \\
        --entry-id a99aa61b
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .shared_styles import SHARED_CSS, esc, fmt, fmt_model, badge
from .generate_error_analysis_pages import (
    generate_error_analysis_pages,
    generate_folder_name,
)

DASHBOARD_ROOT = Path(__file__).parent.parent.parent  # demo/agent-eval-dashboard
DETAILS_BASE = DASHBOARD_ROOT / "leaderboard" / "details"


# ---------------------------------------------------------------------------
# Data extraction (same as original)
# ---------------------------------------------------------------------------


def _parse_sv_string(sv_str: str) -> dict:
    """Parse a SubGoalValidationResult repr string into the standard sv dict."""
    m = re.search(r"is_completed=(True|False)", sv_str)
    is_completed = m.group(1) == "True" if m else False

    m = re.search(r'details="([^"]+)"', sv_str)
    if not m:
        m = re.search(r"details='([^']+)'", sv_str)
    details = m.group(1) if m else ""

    m = re.search(r'(?:GRADE: [CI][^\'"]*)', sv_str)
    explanation = ""
    if m:
        start = sv_str.rfind('"', 0, m.start())
        if start == -1:
            start = sv_str.rfind("'", 0, m.start())
        if start != -1:
            snippet = sv_str[start + 1 : m.end()]
            explanation = snippet[:2000]

    return {
        "subgoal": {"details": details},
        "is_completed": is_completed,
        "explanation": explanation,
    }


def _strip_sample(s: dict) -> dict:
    """Return sample with prompt/extra-explanation fields removed."""
    svs = []
    for sv in s.get("metrics", {}).get("subgoal_validations", []):
        if isinstance(sv, str):
            svs.append(_parse_sv_string(sv))
        else:
            exps = sv.get("explanations", [])
            explanation = exps[1] if len(exps) > 1 else (exps[0] if exps else "")
            svs.append(
                {
                    "subgoal": sv.get("subgoal", {}),
                    "is_completed": sv.get("is_completed", False),
                    "explanation": explanation,
                }
            )
    return {
        "sample_id": s["sample_id"],
        "status": s.get("status", ""),
        "total_turns": s.get("total_turns", 0),
        "trajectory": s.get("trajectory", []),
        "metrics": {
            "auc_score": s.get("metrics", {}).get("auc_score"),
            "ppt_score": s.get("metrics", {}).get("ppt_score"),
            "progress_rates": s.get("metrics", {}).get("progress_rates", []),
            "subgoal_validations": svs,
        },
    }


def load_trial(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "trial_id": raw.get("trial_id", 0),
        "timestamp": raw.get("timestamp", ""),
        "total_samples": raw.get("total_samples", 0),
        "successful": raw.get("successful", 0),
        "failed": raw.get("failed", 0),
        "samples": [_strip_sample(s) for s in raw.get("samples", [])],
    }


def _avg(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------


def _build_summary_page(
    entry: dict,
    dataset_name: str,
    trials: list,
    folder_name: str,
    has_error_analysis: bool = False,
) -> str:

    model = fmt_model(entry.get("agent_model", ""))
    agent = entry.get("agent", "—")
    exp_ts = entry.get("experiment_timestamp", "—")
    n_trials = entry.get("n_trials", len(trials))
    max_turns = entry.get("max_turns", "—")
    persona = entry.get("user_proxy_persona") or "—"
    up_model = entry.get("user_proxy_model", "—")
    entry_id = entry["id"]

    m = entry.get("metrics", {})
    auc_avg = fmt(m.get("auc_score", {}).get("avg_max"))
    ppt_avg = fmt(m.get("ppt_score", {}).get("avg_max"))
    max_pr = fmt(m.get("max_progress_rate", {}).get("avg_max"))
    mean_pr = fmt(m.get("mean_progress_rate", {}).get("avg_max"))
    passatk = m.get("pass_at_k") or m.get("pass@k") or {}
    pass_avg = fmt(passatk.get("avg")) if passatk else "—"
    pass_k = passatk.get("k", "k") if passatk else "k"
    passhatk = m.get("pass_hat_k") or m.get("pass^k") or {}
    passh_avg = fmt(passhatk.get("avg")) if passhatk else "—"
    passh_k = passhatk.get("k", "k") if passhatk else "k"

    # Error analysis button (will be placed below metrics)
    error_analysis_button = ""
    if has_error_analysis:
        error_analysis_button = f"""
    <div class="error-analysis-cta">
      <a class="error-analysis-btn" href="../../error_analysis/{folder_name}/index.html">
        View Error Analysis
      </a>
    </div>"""

    # Per-trial rows
    trial_rows = ""
    for t in sorted(trials, key=lambda x: x["trial_id"]):
        tid = t["trial_id"]
        trial_rows += f"""
        <tr>
          <td class="num">Trial {tid}</td>
          <td class="num">{t["total_samples"]}</td>
          <td class="num">{t["successful"]}</td>
          <td class="num">{t["failed"]}</td>
          <td><a class="link-btn" href="trial_{tid}.html">View &rsaquo;</a></td>
        </tr>"""

    # Per-sample aggregate rows
    sample_ids = [s["sample_id"] for s in trials[0]["samples"]] if trials else []
    auc_mps = m.get("auc_score", {}).get("max_per_sample", {})
    ppt_mps = m.get("ppt_score", {}).get("max_per_sample", {})
    maxpr_mps = m.get("max_progress_rate", {}).get("max_per_sample", {})
    meanpr_mps = m.get("mean_progress_rate", {}).get("per_sample", {})
    passk_ps = (m.get("pass_at_k") or m.get("pass@k") or {}).get("per_sample", {})
    passhk_ps = (m.get("pass_hat_k") or m.get("pass^k") or {}).get("per_sample", {})

    sample_rows = ""
    for sid in sample_ids:
        sid_pass, sid_total = 0, 0
        subgoal_counts: dict = {}
        subgoal_totals: dict = {}
        for t in trials:
            for s in t["samples"]:
                if s["sample_id"] != sid:
                    continue
                sid_total += 1
                for sv in s["metrics"]["subgoal_validations"]:
                    k = sv["subgoal"].get("details", "")[:60]
                    subgoal_totals[k] = subgoal_totals.get(k, 0) + 1
                    if sv["is_completed"]:
                        subgoal_counts[k] = subgoal_counts.get(k, 0) + 1
                if all(sv["is_completed"] for sv in s["metrics"]["subgoal_validations"]):
                    sid_pass += 1

        n_sg = len(subgoal_totals)
        sg_passed = sum(
            1 for k in subgoal_totals if (subgoal_counts.get(k, 0) / subgoal_totals[k]) >= 0.5
        )
        pass_rate = f"{sid_pass}/{sid_total}" if sid_total else "—"
        sample_rows += f"""
        <tr>
          <td style="font-family:var(--font-mono);font-size:12px;color:var(--text-2)">{esc(sid)}</td>
          <td>{badge(auc_mps.get(sid))}</td>
          <td>{badge(ppt_mps.get(sid))}</td>
          <td>{badge(maxpr_mps.get(sid))}</td>
          <td>{badge(meanpr_mps.get(sid))}</td>
          <td>{badge(passk_ps.get(sid))}</td>
          <td>{badge(passhk_ps.get(sid))}</td>
          <td class="num">{pass_rate}</td>
          <td class="num">{sg_passed}/{n_sg} subgoals (majority)</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(model)} — Experiment Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}

/* Error Analysis CTA Button */
.error-analysis-cta {{
  margin: 32px 0;
  display: flex;
  justify-content: flex-start;
}}

.error-analysis-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 16px 48px;
  background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.02em;
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25);
  transition: all 0.2s ease;
  border: 2px solid transparent;
}}

.error-analysis-btn:hover {{
  background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
  box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35);
  transform: translateY(-2px);
}}
</style>
</head>
<body>

<nav class="nav">
  <div class="container nav-inner">
    <a class="nav-brand" href="../../index.html">
      Agentic AI Metrics
      <span class="nav-divider">/</span>
      <span class="nav-section">Leaderboard</span>
    </a>
  </div>
</nav>

<div class="container">
  <div class="breadcrumb">
    <a href="../../index.html">Leaderboard</a>
    <span class="sep">/</span>
    <span>{esc(dataset_name)}</span>
    <span class="sep">/</span>
    <span>{esc(model)}</span>
  </div>

  <div class="page-header">
    <div class="page-eyebrow">Experiment Report</div>
    <h1>{esc(model)}</h1>
    <div class="subtitle">{esc(agent)}</div>
    <div class="meta-row">
      <span class="meta-pill">{n_trials} Trials</span>
      <span class="meta-pill">Max {max_turns} Turns</span>
      <span class="meta-pill">Persona: {esc(str(persona))}</span>
      <span class="meta-pill">User proxy: {esc(up_model)}</span>
      <span class="meta-pill">{esc(exp_ts)}</span>
    </div>
    <div class="metrics-row">
      <div class="metric-card">
        <div class="mc-label">MaxAUC@k</div>
        <div class="mc-value">{auc_avg}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">MaxPPT@k</div>
        <div class="mc-value">{ppt_avg}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Pass@{pass_k}</div>
        <div class="mc-value">{pass_avg}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Pass^{passh_k}</div>
        <div class="mc-value">{passh_avg}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">MaxProg@k</div>
        <div class="mc-value">{max_pr}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">MeanProg@k</div>
        <div class="mc-value">{mean_pr}</div>
      </div>
    </div>
  </div>

  {error_analysis_button}

  <div class="section-header">
    <h2>Trial Results</h2>
    <span class="section-sub">{len(trials)} trials — click View to inspect trajectories</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Trial</th><th>Samples</th><th>Successful</th><th>Failed</th><th></th>
      </tr></thead>
      <tbody>{trial_rows}</tbody>
    </table>
  </div>

  <div class="section-header">
    <h2>Per-Sample Performance</h2>
    <span class="section-sub">Aggregated across all {len(trials)} trials</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Sample</th><th>MaxAUC@k</th><th>MaxPPT@k</th>
        <th>MaxProg@k</th><th>MeanProg@k</th>
        <th>Pass@k</th><th>Pass^k</th>
        <th>Pass Rate</th><th>Subgoals</th>
      </tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
  </div>
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Trial page CSS
# ---------------------------------------------------------------------------

_TRIAL_PAGE_CSS = """
/* Two-panel layout */
.layout {
  display: grid;
  grid-template-columns: 264px 1fr;
  gap: 0;
  height: calc(100vh - 96px);
  overflow: hidden;
}
.sidebar {
  border-right: 1px solid var(--border);
  background: var(--surface);
  overflow-y: auto;
}
.sidebar-header {
  padding: 12px 16px 10px;
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-3); border-bottom: 1px solid var(--border-sub);
  position: sticky; top: 0; background: var(--surface); z-index: 1;
}
.sample-item {
  padding: 11px 16px; border-bottom: 1px solid var(--border-sub);
  cursor: pointer; transition: background 0.08s;
}
.sample-item:hover { background: var(--bg); }
.sample-item.active {
  background: var(--brand-bg);
  border-left: 2px solid var(--brand);
  padding-left: 14px;
}
.sample-item .s-id {
  font-family: var(--font-mono);
  font-size: 12px; color: var(--text-1); margin-bottom: 4px;
}
.sample-item .s-meta { display: flex; gap: 8px; align-items: center; }
.sample-item .s-auc  { font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
.sample-item .s-turns { font-size: 11px; color: var(--text-3); }
.sample-item .s-sg    { font-size: 11px; color: var(--text-3); }

/* Main panel */
.main-panel { overflow-y: auto; padding: 24px 28px; background: var(--bg); }

/* Sample detail header */
.sample-detail-header {
  margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
}
.sample-detail-header h2 {
  font-size: 14px; font-weight: 600; color: var(--text-1); margin-bottom: 8px;
  font-family: var(--font-mono);
}
.sample-stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }

/* Progress bar */
.progress-section { margin-bottom: 24px; }
.progress-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-2); font-weight: 600; margin-bottom: 8px;
}
.progress-track {
  height: 8px; background: var(--border); border-radius: 4px;
  display: flex; gap: 2px;
}
.progress-seg { flex: 1; border-radius: 2px; }

/* Conversation */
.conversation-section h3 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-2); font-weight: 600; margin-bottom: 14px;
  padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.turn-block { margin-bottom: 20px; }
.turn-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--border); font-weight: 600; margin-bottom: 6px;
}
.msg-block {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 6px; overflow: hidden;
}
.msg-role-bar {
  padding: 5px 14px; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border-sub);
}
.role-user      { background: var(--bg); color: var(--text-3); }
.role-agent     { background: var(--brand-bg); color: var(--brand); }
.role-tool-call { background: var(--bg); color: var(--text-2); }
.role-tool-resp { background: var(--bg); color: var(--text-3); }
.msg-body {
  padding: 12px 14px; font-size: 13px; line-height: 1.65;
  color: var(--text-1); white-space: pre-wrap; word-break: break-word;
}
.msg-code {
  padding: 12px 14px; font-family: var(--font-mono);
  font-size: 12px; line-height: 1.55; color: var(--text-1);
  overflow-x: auto; white-space: pre; background: var(--bg);
}
.msg-body code { font-family: var(--font-mono); font-size: 0.9em; }
details summary {
  padding: 7px 14px; font-size: 12px; font-weight: 500; color: var(--text-2);
  cursor: pointer; list-style: none;
}
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "+ "; color: var(--text-3); }
details[open] summary::before { content: "− "; }
details pre {
  padding: 0 14px 12px; font-family: var(--font-mono);
  font-size: 12px; line-height: 1.5; color: var(--text-2);
  overflow-x: auto; white-space: pre;
}

/* Subgoal table */
.subgoals-section { margin-top: 28px; }
.subgoals-section h3 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-2); font-weight: 600; margin-bottom: 12px;
  padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.sg-table {
  width: 100%; border-collapse: collapse; font-size: 13px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
}
.sg-table th {
  padding: 9px 14px; text-align: left; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-2);
  background: var(--bg); border-bottom: 1px solid var(--border);
}
.sg-table td { padding: 11px 14px; border-bottom: 1px solid var(--border-sub); vertical-align: top; }
.sg-table tr:last-child td { border-bottom: none; }
.sg-table .sg-detail { font-size: 13px; color: var(--text-1); line-height: 1.5; }
.sg-table .sg-exp    { font-size: 12px; color: var(--text-2); line-height: 1.5; margin-top: 4px; }
.sg-status-complete   { font-size: 12px; font-weight: 600; color: var(--score-hi); white-space: nowrap; }
.sg-status-incomplete { font-size: 12px; font-weight: 600; color: var(--score-lo); white-space: nowrap; }

/* Placeholder */
.placeholder {
  display: flex; align-items: center; justify-content: center;
  height: 100%; color: var(--text-3); font-size: 13px;
}
"""


def _build_trial_page(
    entry: dict,
    dataset_name: str,
    trial: dict,
    n_trials: int,
    folder_name: str,
    summary_path: str = "index.html",
) -> str:
    model = fmt_model(entry.get("agent_model", ""))
    agent = entry.get("agent", "—")
    exp_ts = entry.get("experiment_timestamp", "—")
    tid = trial["trial_id"]

    prev_link = (
        "" if tid == 0 else f'<a class="link-btn" href="trial_{tid - 1}.html">← Trial {tid - 1}</a>'
    )
    next_link = (
        ""
        if tid >= n_trials - 1
        else f'<a class="link-btn" href="trial_{tid + 1}.html">Trial {tid + 1} →</a>'
    )

    samples_json = json.dumps(trial["samples"], ensure_ascii=False, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(model)} — Trial {tid}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}
{_TRIAL_PAGE_CSS}
</style>
</head>
<body>

<nav class="nav" style="position:sticky;top:0;z-index:100">
  <div style="max-width:100%;padding:0 20px;display:flex;align-items:center;justify-content:space-between;width:100%">
    <a class="nav-brand" href="../../index.html">
      Agentic AI Metrics
      <span class="nav-divider">/</span>
      <span class="nav-section">Leaderboard</span>
    </a>
    <div class="breadcrumb" style="padding:0;margin:0">
      <a href="../../index.html">Leaderboard</a>
      <span class="sep">/</span>
      <span>{esc(dataset_name)}</span>
      <span class="sep">/</span>
      <a href="{summary_path}">{esc(model)}</a>
      <span class="sep">/</span>
      <span>Trial {tid}</span>
    </div>
  </div>
</nav>

<div style="background:var(--surface);border-bottom:1px solid var(--border);padding:10px 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <span style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-3)">Trial {tid} of {n_trials - 1}</span>
      <span style="margin-left:12px;font-size:12px;color:var(--text-2)">{esc(agent)} &nbsp;·&nbsp; {trial["successful"]}/{trial["total_samples"]} successful &nbsp;·&nbsp; {esc(exp_ts)}</span>
    </div>
    <div style="display:flex;gap:6px">
      {prev_link}
      {next_link}
    </div>
  </div>
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">Samples <span style="color:var(--text-3);font-weight:400">{trial["total_samples"]}</span></div>
    <div id="sample-list"></div>
  </div>

  <div class="main-panel" id="main-panel">
    <div class="placeholder">Select a sample to view its trajectory.</div>
  </div>
</div>

<script>
const SAMPLES = {samples_json};

function esc(s) {{
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}}

function badge(v) {{
  if (v == null) return '<span class="score-na">—</span>';
  const n = parseFloat(v);
  const cls = n >= 0.75 ? 'score-hi' : n >= 0.50 ? 'score-mid' : 'score-lo';
  return `<span class="score-val ${{cls}}">${{n.toFixed(3)}}</span>`;
}}

function renderSidebar() {{
  const sorted = SAMPLES.slice().sort((a,b) =>
    (b.metrics.auc_score ?? -1) - (a.metrics.auc_score ?? -1));
  const container = document.getElementById('sample-list');
  container.innerHTML = sorted.map((s, idx) => {{
    const svs = s.metrics.subgoal_validations || [];
    const passed = svs.filter(sv => sv.is_completed).length;
    const auc = s.metrics.auc_score;
    const cls = auc != null ? (auc >= 0.75 ? 'complete' : auc >= 0.50 ? '' : 'incomplete') : 'neutral';
    return `<div class="sample-item" data-id="${{esc(s.sample_id)}}" onclick="selectSample('${{esc(s.sample_id)}}', this)">
      <div class="s-id">${{esc(s.sample_id)}}</div>
      <div class="s-meta">
        <span class="s-auc ${{cls}}">${{auc != null ? auc.toFixed(3) : '—'}}</span>
        <span class="s-turns">${{s.total_turns}} turns</span>
        <span class="s-sg">${{passed}}/${{svs.length}} subgoals</span>
      </div>
    </div>`;
  }}).join('');
}}

function selectSample(sampleId, el) {{
  document.querySelectorAll('.sample-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');
  const s = SAMPLES.find(x => x.sample_id === sampleId);
  if (!s) return;
  renderConversation(s);
}}

function renderConversation(s) {{
  const panel = document.getElementById('main-panel');
  const svs = s.metrics.subgoal_validations || [];
  const passed = svs.filter(sv => sv.is_completed).length;
  const rates = s.metrics.progress_rates || [];

  const maxRate = Math.max(...rates, 0.001);
  const segs = rates.map(r => {{
    const col = r >= 0.75 ? '#059669' : r >= 0.50 ? '#d97706' : r > 0 ? '#0070f3' : '#e3e6ef';
    return `<div class="progress-seg" style="background:${{col}}" title="Turn: ${{r.toFixed(3)}}"></div>`;
  }}).join('');

  let convHtml = '';
  (s.trajectory || []).forEach((turn, ti) => {{
    convHtml += `<div class="turn-block"><div class="turn-label">Turn ${{ti + 1}}</div>`;
    turn.forEach(msg => {{
      if (msg.role === 'user') {{
        convHtml += `<div class="msg-block">
          <div class="msg-role-bar role-user">User</div>
          <div class="msg-body">${{esc(msg.content || '')}}</div>
        </div>`;
      }} else if (msg.role === 'agent') {{
        if (msg.tool_calls && msg.tool_calls.length) {{
          msg.tool_calls.forEach(tc => {{
            const args = JSON.stringify(tc.arguments || {{}}, null, 2);
            convHtml += `<div class="msg-block">
              <div class="msg-role-bar role-tool-call">Tool Call — ${{esc(tc.name)}}</div>
              <div class="msg-code">${{esc(args)}}</div>
            </div>`;
          }});
        }}
        if (msg.content) {{
          convHtml += `<div class="msg-block">
            <div class="msg-role-bar role-agent">Agent</div>
            <div class="msg-body">${{esc(msg.content)}}</div>
          </div>`;
        }}
      }} else if (msg.role === 'tool') {{
        let pretty;
        try {{ pretty = JSON.stringify(JSON.parse(msg.content || 'null'), null, 2); }}
        catch {{ pretty = msg.content || ''; }}
        convHtml += `<div class="msg-block">
          <div class="msg-role-bar role-tool-resp">Tool Response</div>
          <details><summary>View payload</summary><pre>${{esc(pretty)}}</pre></details>
        </div>`;
      }}
    }});
    convHtml += `</div>`;
  }});

  let sgHtml = svs.map((sv, i) => {{
    const done = sv.is_completed;
    return `<tr>
      <td style="color:#9aa5b4;font-size:0.75rem;width:32px;text-align:center">${{i+1}}</td>
      <td><div class="sg-detail">${{esc(sv.subgoal.details || '')}}</div>
          ${{sv.explanation ? `<div class="sg-exp">${{esc(sv.explanation)}}</div>` : ''}}
      </td>
      <td><span class="${{done ? 'sg-status-complete' : 'sg-status-incomplete'}}">${{done ? 'Complete' : 'Incomplete'}}</span></td>
    </tr>`;
  }}).join('');

  panel.innerHTML = `
    <div class="sample-detail-header">
      <h2>${{esc(s.sample_id)}}</h2>
      <div class="sample-stats">
        <span class="meta-pill">AUC: ${{s.metrics.auc_score != null ? s.metrics.auc_score.toFixed(3) : '—'}}</span>
        <span class="meta-pill">PPT: ${{s.metrics.ppt_score != null ? s.metrics.ppt_score.toFixed(3) : '—'}}</span>
        <span class="meta-pill">${{s.total_turns}} turns</span>
        <span class="meta-pill">${{passed}}/${{svs.length}} subgoals complete</span>
        <span class="meta-pill">Status: ${{esc(s.status)}}</span>
      </div>
      <div class="progress-section">
        <div class="progress-label">Progress per turn (AUC)</div>
        <div class="progress-track" style="height:10px">${{segs}}</div>
        <div style="display:flex;gap:4px;margin-top:4px">
          ${{rates.map((r,i) => `<div style="flex:1;font-size:0.62rem;color:#9aa5b4;text-align:center">${{r.toFixed(2)}}</div>`).join('')}}
        </div>
      </div>
    </div>

    <div class="conversation-section">
      <h3>Conversation Trajectory</h3>
      ${{convHtml}}
    </div>

    <div class="subgoals-section">
      <h3>Subgoal Evaluations</h3>
      <table class="sg-table">
        <thead><tr><th>#</th><th>Subgoal</th><th>Result</th></tr></thead>
        <tbody>${{sgHtml}}</tbody>
      </table>
    </div>`;
}}

renderSidebar();
if (SAMPLES.length) {{
  const sorted = SAMPLES.slice().sort((a,b) =>
    (b.metrics.auc_score ?? -1) - (a.metrics.auc_score ?? -1));
  const first = document.querySelector(`[data-id="${{sorted[0].sample_id}}"]`);
  if (first) selectSample(sorted[0].sample_id, first);
}}
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_detail_pages(
    entry: dict,
    dataset_name: str,
    dataset_key: str,
    generate_error_pages: bool = True,
) -> Path:
    """
    Generate detail pages for an entry with improved folder naming.

    Folder format: {dataset}_{model}_{persona}_{id}
    Example: tau2bench_airline_gpt4_expert_a99aa61b

    Returns:
        Path to the summary page
    """
    source = Path(entry["source_folder"])
    folder_name = generate_folder_name(entry, dataset_key)
    out_dir = DETAILS_BASE / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find trial files
    trial_files = sorted(
        source.glob("trial_*_results.json"),
        key=lambda p: int(re.search(r"trial_(\d+)_", p.name).group(1)),
    )

    if not trial_files:
        print(f"  [warn] No trial_N_results.json found in {source}")
        return out_dir / "index.html"

    print(f"  Reading {len(trial_files)} trial file(s) from {source.name}...")
    trials = []
    for tf in trial_files:
        print(f"    trial {tf.name} ...", end="", flush=True)
        t = load_trial(tf)
        trials.append(t)
        print(" done")

    # Check for error analysis
    has_error_analysis = (source / "error_analysis.pkl").exists()

    # Generate error analysis pages if available
    if generate_error_pages and has_error_analysis:
        generate_error_analysis_pages(entry, dataset_name, source)

    # Summary page
    summary_html = _build_summary_page(entry, dataset_name, trials, folder_name, has_error_analysis)
    summary_path = out_dir / "index.html"
    summary_path.write_text(summary_html, encoding="utf-8")
    print(f"  Summary   → {summary_path}")

    # Per-trial pages
    for t in trials:
        trial_html = _build_trial_page(entry, dataset_name, t, len(trial_files), folder_name)
        tp = out_dir / f"trial_{t['trial_id']}.html"
        tp.write_text(trial_html, encoding="utf-8")
        print(f"  Trial {t['trial_id']:>2d}   → {tp.name}")

    return summary_path
