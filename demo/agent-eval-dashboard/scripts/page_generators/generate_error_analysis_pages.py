"""
Generate static HTML error analysis pages with improved interactive controls.

This version uses proper HTML <select> dropdowns with multiple selection support
instead of chip-based selection for better usability.

Features:
- Multi-select dropdowns for samples and trace pairs
- Cross-trace pairwise comparison dot plot with error bars
- Error category summary table (cluster × sample matrix)
- Binary heatmap (Subgoal × Judge) for selected pairs
- Judge details with expandable sections

Usage:
    from page_generators.generate_error_analysis_pages import generate_error_analysis_pages
    generate_error_analysis_pages(entry, dataset_name, source_folder)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_inspect.tools.error_analysis.statistic_analysis import StatisticAnalysis

from .shared_styles import SHARED_CSS, esc, fmt, fmt_model, slugify

# Add src to path to allow pickle to load agent_inspect modules
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import StatisticAnalysis directly to avoid client dependencies
# (which require backoff module that may not be installed)


DASHBOARD_ROOT = Path(__file__).parent.parent.parent  # demo/agent-eval-dashboard
ERROR_ANALYSIS_BASE = DASHBOARD_ROOT / "leaderboard" / "error_analysis"


# ---------------------------------------------------------------------------
# Data extraction and statistical analysis
# ---------------------------------------------------------------------------


def match_to_int(completion: str) -> int:
    """
    Extract grade from judge explanation using official pattern.
    Pattern matches "GRADE: C" or "GRADE: I" (case-insensitive).
    Returns 1 for Complete, 0 for Incomplete, raises ValueError for invalid/missing grade.
    """
    if not completion or completion == "DUMMY STRING":
        raise ValueError("Empty or dummy completion")

    # Official pattern from agent_inspect.metrics.constants
    # Matches: GRADE: C or GRADE: I (case-insensitive, with optional whitespace)
    pattern = r"(?i)GRADE\s*:\s*([CPI])"
    match = re.search(pattern, completion)

    if not match:
        raise ValueError("Could not find GRADE pattern in completion")

    grade = match.group(1).upper()
    if grade == "C":
        return 1  # Complete
    elif grade == "I":
        return 0  # Incomplete
    elif grade == "P":
        raise ValueError("Partial grade 'P' not supported in binary scoring")
    else:
        raise ValueError(f"Invalid grade: {grade}")


def load_error_analysis_data(pkl_path: Path) -> Tuple[List[Any], Any]:
    """Load error analysis data from pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        return data
    else:
        raise ValueError(f"Unexpected data format in {pkl_path}")


def extract_error_data(
    error_analysis_data_samples: List[Any], error_analysis_result: Any
) -> Dict[str, Any]:
    """Extract and organize error data for visualization.

    This function mimics the Streamlit ResultHandler.prepare_dataframes() logic
    to ensure the error summary table matches the Streamlit UI.
    """

    mean_data = []
    full_data = []

    # Track subgoal indices per (agent_run_id, sample_id) to maintain consistency
    sample_subgoal_dict = defaultdict(dict)

    # First pass: Compute statistics using StatisticAnalysis (same as Streamlit)
    for sample in error_analysis_data_samples:
        agent_run_id = sample.agent_run_id if sample.agent_run_id is not None else "0"
        sample_id = sample.data_sample_id
        key = (str(agent_run_id), sample_id)

        # Use StatisticAnalysis to compute statistics (same as Streamlit launch_ui)
        stat_result = StatisticAnalysis.compute_statistic_analysis_result(sample)

        mean_data.append(
            {
                "agent_run_id": str(agent_run_id),
                "sample_idx": sample_id,
                "SB_token_prob_mean": stat_result.judge_expectation
                if stat_result.judge_expectation is not None
                else 0.0,
                "SB_token_prob_SD_aggregated": stat_result.judge_std
                if stat_result.judge_std is not None
                else 0.0,
            }
        )

    # Second pass: Process clustered errors (WITH cluster_label and final_error_type)
    # This matches ResultHandler.prepare_dataframes() lines 62-71
    if error_analysis_result and hasattr(
        error_analysis_result, "analyzed_validations_clustered_by_errors"
    ):
        for (
            cluster_label,
            analyzed_validations,
        ) in error_analysis_result.analyzed_validations_clustered_by_errors.items():
            for av in analyzed_validations:
                if av is None or av.subgoal_validation is None:
                    continue

                agent_run_id = str(av.agent_run_id if av.agent_run_id is not None else "0")
                sample_id = av.data_sample_id
                subgoal_detail = (
                    av.subgoal_validation.sub_goal.details if av.subgoal_validation.sub_goal else ""
                )

                # Track subgoal index
                key = (agent_run_id, sample_id)
                if subgoal_detail not in sample_subgoal_dict[key]:
                    subgoal_idx = len(sample_subgoal_dict[key]) + 1
                    sample_subgoal_dict[key][subgoal_detail] = subgoal_idx
                else:
                    subgoal_idx = sample_subgoal_dict[key][subgoal_detail]

                row = {
                    "agent_run_id": agent_run_id,
                    "sample_idx": sample_id,
                    "subgoal_idx": subgoal_idx,
                    "subgoal_detail": subgoal_detail,
                    "judge_model_input": av.subgoal_validation.prompt_sent_to_llmj
                    if hasattr(av.subgoal_validation, "prompt_sent_to_llmj")
                    else "",
                    "cluster_label": cluster_label,
                    "final_error_type": av.base_error if hasattr(av, "base_error") else "",
                }

                # Extract judge scores
                explanations = (
                    av.subgoal_validation.explanations[1:]
                    if len(av.subgoal_validation.explanations) > 1
                    else []
                )
                for j, exp in enumerate(explanations):
                    if exp == "DUMMY STRING":
                        continue
                    row[f"pred_{j}"] = exp
                    try:
                        score = match_to_int(exp)
                        row[f"pred_{j}_score"] = float(score)
                    except ValueError:
                        # If pattern match fails, default to 0.0
                        row[f"pred_{j}_score"] = 0.0

                full_data.append(row)

    # Third pass: Process completed subgoals (WITHOUT cluster_label)
    # This matches ResultHandler.prepare_dataframes() lines 74-81
    if error_analysis_result and hasattr(error_analysis_result, "completed_subgoal_validations"):
        for cv in error_analysis_result.completed_subgoal_validations:
            if cv is None or cv.subgoal_validation is None:
                continue

            agent_run_id = str(cv.agent_run_id if cv.agent_run_id is not None else "0")
            sample_id = cv.data_sample_id
            subgoal_detail = (
                cv.subgoal_validation.sub_goal.details if cv.subgoal_validation.sub_goal else ""
            )

            # Track subgoal index
            key = (agent_run_id, sample_id)
            if subgoal_detail not in sample_subgoal_dict[key]:
                subgoal_idx = len(sample_subgoal_dict[key]) + 1
                sample_subgoal_dict[key][subgoal_detail] = subgoal_idx
            else:
                subgoal_idx = sample_subgoal_dict[key][subgoal_detail]

            row = {
                "agent_run_id": agent_run_id,
                "sample_idx": sample_id,
                "subgoal_idx": subgoal_idx,
                "subgoal_detail": subgoal_detail,
                "judge_model_input": cv.subgoal_validation.prompt_sent_to_llmj
                if hasattr(cv.subgoal_validation, "prompt_sent_to_llmj")
                else "",
            }

            # Extract judge scores
            explanations = (
                cv.subgoal_validation.explanations[1:]
                if len(cv.subgoal_validation.explanations) > 1
                else []
            )
            for j, exp in enumerate(explanations):
                if exp == "DUMMY STRING":
                    continue
                row[f"pred_{j}"] = exp
                try:
                    score = match_to_int(exp)
                    row[f"pred_{j}_score"] = float(score)
                except ValueError:
                    # If pattern match fails, default to 0.0
                    row[f"pred_{j}_score"] = 0.0

            full_data.append(row)

    # Reassign subgoal_idx consistently across all agent_run_ids
    # This matches ResultHandler.reassign_subgoal_idx() logic (lines 90-108)
    # For each sample_idx, sort subgoal_details alphabetically and reassign indices 1, 2, 3...
    if full_data:
        # Group by sample_idx and collect unique subgoal_details
        sample_subgoals = defaultdict(set)
        for row in full_data:
            sample_subgoals[row["sample_idx"]].add(row["subgoal_detail"])

        # Sort subgoal_details alphabetically for each sample and create mapping
        subgoal_idx_map = {}
        for sample_idx, subgoal_details in sample_subgoals.items():
            sorted_details = sorted(subgoal_details)
            subgoal_idx_map[sample_idx] = {
                detail: idx + 1 for idx, detail in enumerate(sorted_details)
            }

        # Apply the mapping to reassign subgoal_idx
        for row in full_data:
            row["subgoal_idx"] = subgoal_idx_map[row["sample_idx"]][row["subgoal_detail"]]

    # Get unique agent runs and samples from full_data (which now includes all agent runs)
    # Sort agent_runs numerically to match Streamlit's trace mapping
    agent_runs = sorted(set(row["agent_run_id"] for row in full_data), key=lambda x: int(x))
    sample_ids = sorted(set(row["sample_idx"] for row in full_data))

    # Count number of judges from first sample with data
    num_judges = 0
    if full_data:
        judge_keys = [
            k for k in full_data[0].keys() if k.startswith("pred_") and k.endswith("_score")
        ]
        num_judges = len(judge_keys)

    # Summary statistics: Count errors and completions from full_data
    error_rows = [row for row in full_data if row.get("cluster_label")]
    completed_rows = [row for row in full_data if not row.get("cluster_label")]
    cluster_labels = set(row["cluster_label"] for row in error_rows)

    total_subgoals = len(full_data)
    completed_count = len(completed_rows)
    error_count = len(error_rows)
    num_error_types = len(cluster_labels)

    return {
        "mean_data": mean_data,
        "full_data": full_data,
        "agent_runs": agent_runs,
        "sample_ids": sample_ids,
        "num_judges": num_judges,
        "summary": {
            "num_samples": len(sample_ids),
            "num_traces": len(agent_runs),
            "total_subgoals": total_subgoals,
            "completed_count": completed_count,
            "error_count": error_count,
            "num_error_types": num_error_types,
        },
    }


def build_error_summary_table(error_data: Dict[str, Any]) -> str:
    """Build the error category summary table HTML.

    This function mimics the Streamlit error_summary_matrix_tab_simple() logic
    to ensure the table matches the Streamlit UI.
    """
    full_data = error_data["full_data"]
    sample_ids = error_data["sample_ids"]
    agent_runs = error_data["agent_runs"]

    # Filter to only rows with cluster_label (errors)
    # Match Streamlit logic: lines 60-66 of consistency_viz.py
    error_rows = [
        row for row in full_data if row.get("cluster_label") and row["cluster_label"] != ""
    ]

    # Filter out rows where cluster_label contains BOTH "no" AND "error"
    error_rows = [
        row
        for row in error_rows
        if not ("no" in row["cluster_label"].lower() and "error" in row["cluster_label"].lower())
    ]

    if not error_rows:
        return ""

    # Build trace map (agent_run_id -> trace_N)
    trace_map = {run_id: f"trace_{i + 1}" for i, run_id in enumerate(agent_runs)}

    # Get unique cluster labels
    cluster_labels = sorted(set(row["cluster_label"] for row in error_rows))

    # Build table header
    header_cells = "".join(f"<th>sample_{esc(str(sid))}</th>" for sid in sample_ids)

    # Build table rows (match Streamlit logic: lines 82-107)
    rows = []
    for cluster_label in cluster_labels:
        # Count occurrences per (sample, trace) for this cluster
        sample_traces = defaultdict(set)  # Use set to avoid duplicates
        total_cases = 0

        for row in error_rows:
            if row["cluster_label"] == cluster_label:
                sample_idx = row["sample_idx"]
                agent_run_id = row["agent_run_id"]
                trace = trace_map.get(agent_run_id, "trace_1")
                sample_traces[sample_idx].add(trace)
                total_cases += 1  # Count every occurrence (match Streamlit line 101)

        # Build row cells with truncation for cells with many traces
        cells = []
        MAX_VISIBLE_TRACES = 5  # Show first 5 traces, then "+ N more"

        for sid in sample_ids:
            # Sort traces numerically (trace_1, trace_2, ..., trace_10, trace_11, ...)
            traces = sorted(
                sample_traces.get(sid, []),
                key=lambda t: int(t.split("_")[1]) if "_" in t else 0,
            )
            if traces:
                cell_class = ' class="trace-cell"'
                if len(traces) <= MAX_VISIBLE_TRACES:
                    # Show all traces if count is within limit
                    content = ", ".join(traces)
                else:
                    # Show first N traces, then "+ N more" button
                    visible_traces = ", ".join(traces[:MAX_VISIBLE_TRACES])
                    hidden_traces = ", ".join(traces[MAX_VISIBLE_TRACES:])
                    more_count = len(traces) - MAX_VISIBLE_TRACES
                    cell_id = (
                        f"cell_{cluster_label.replace(' ', '_').replace('/', '_')}_{sid}".replace(
                            "-", "_"
                        )
                    )
                    content = f"""<span class="trace-list-visible">{visible_traces}</span><span class="trace-list-hidden" id="hidden_{cell_id}" style="display:none">, {hidden_traces}</span><button class="trace-expand-btn" onclick="toggleTraces('{cell_id}')" id="btn_{cell_id}">+{more_count} more</button>"""
            else:
                cell_class = ""
                content = "–"
            cells.append(f"<td{cell_class}>{content}</td>")

        row_html = f"""<tr><td class="sticky-col">{esc(cluster_label)}</td>{"".join(cells)}<td>{total_cases}</td></tr>"""
        rows.append(row_html)

    if not rows:
        return ""

    return f"""
    <div class="error-summary-container">
      <div class="error-summary-title">Error Category Summary (Columns: sample_idx, Content: trace idx)</div>
      <div class="error-summary-scroll-wrapper">
        <table class="error-summary-table">
          <thead>
            <tr>
              <th class="sticky-col">LLM summarized error category</th>
      {header_cells}<th>Total Cases</th></tr></thead><tbody>{"".join(rows)}
          </tbody>
        </table>
      </div>
      <div class="error-summary-warning">
        The whole summary is generated and clustered by LLM, only used for reference and it may not be 100% correct.
        Please refer to the below details for double confirm.
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# HTML Page Builder
# ---------------------------------------------------------------------------
from .css_for_html_page import ERROR_ANALYSIS_CSS

def build_error_analysis_page(
    entry: Dict[str, Any],
    dataset_name: str,
    error_data: Dict[str, Any],
    folder_name: str,
) -> str:
    """Build the main error analysis HTML page with improved controls."""

    model = fmt_model(entry.get("agent_model", ""))
    persona = entry.get("user_proxy_persona", "expert")
    summary = error_data["summary"]

    # Prepare data for JavaScript
    data_json = json.dumps(
        {
            "mean_data": error_data["mean_data"],
            "full_data": error_data["full_data"],
            "agent_runs": error_data["agent_runs"],
            "sample_ids": error_data["sample_ids"],
            "num_judges": error_data["num_judges"],
            "summary": summary,
        },
        ensure_ascii=False,
    )

    # Build error summary table HTML
    error_table_html = build_error_summary_table(error_data)

    # Build sample options
    sample_options = "".join(
        f'<option value="{esc(str(sid))}">{esc(str(sid))}</option>'
        for sid in error_data["sample_ids"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Error Analysis — {esc(model)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Georgia&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{SHARED_CSS}
{ERROR_ANALYSIS_CSS}
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
    <div style="display:flex;gap:8px">
      <a class="link-btn" href="../../details/{folder_name}/index.html">&larr; Back to Details</a>
    </div>
  </div>
</nav>

<div class="main-content">
  <!-- Breadcrumb -->
  <div class="breadcrumb">
    <a href="../../index.html">Leaderboard</a>
    <span class="sep">/</span>
    <span>{esc(dataset_name)}</span>
    <span class="sep">/</span>
    <a href="../../details/{folder_name}/index.html">{esc(model)}</a>
    <span class="sep">/</span>
    <span>Error Analysis</span>
  </div>

  <!-- Page header -->
  <div class="page-header">
    <div class="page-eyebrow">Error Analysis</div>
    <h1>{esc(model)}</h1>
    <div class="subtitle">{esc(dataset_name)} &middot; {esc(persona)} persona</div>
  </div>

  <!-- Summary cards -->
  <div class="summary-cards">
    <div class="summary-card">
      <div class="card-value">{summary["num_samples"]}</div>
      <div class="card-label">Samples</div>
    </div>
    <div class="summary-card">
      <div class="card-value">{summary["num_traces"]}</div>
      <div class="card-label">Traces</div>
    </div>
    <div class="summary-card">
      <div class="card-value">{summary["total_subgoals"]}</div>
      <div class="card-label">Total Subgoals</div>
    </div>
    <div class="summary-card">
      <div class="card-value" style="color:var(--score-hi)">{summary["completed_count"]}</div>
      <div class="card-label">Completed</div>
    </div>
    <div class="summary-card">
      <div class="card-value" style="color:var(--score-lo)">{summary["error_count"]}</div>
      <div class="card-label">Errors</div>
    </div>
    <div class="summary-card">
      <div class="card-value">{summary["num_error_types"]}</div>
      <div class="card-label">Error Types</div>
    </div>
  </div>

  <!-- Error Summary Table -->
  {error_table_html}

  <!-- Dot Plot Pair Selector -->
  <div class="selection-section">
    <div class="selection-title">Filter Pairs for Dot Plot</div>
    <p style="color:var(--text-2);font-size:13px;margin-bottom:16px">Click on pairs to filter the dot plot. Use "Select All" buttons to quickly select all traces for a sample.</p>

    <!-- Search bar -->
    <div style="margin-bottom:16px">
      <input type="text" id="dotplot-pair-search" class="search-input" placeholder="🔍 Search pairs (trace or sample)..." oninput="filterDotPlotPairList(this.value)">
    </div>

    <!-- Action buttons -->
    <div style="display:flex;gap:12px;margin-bottom:16px">
      <button class="btn-secondary" onclick="selectAllDotPlotPairs()">Select All</button>
      <button class="btn-secondary" onclick="clearAllDotPlotPairs()">Clear All</button>
    </div>

    <!-- Pair list with sample grouping -->
    <div class="pair-list-container">
      <div class="pair-list" id="dotplot-pair-list">
        <!-- List will be populated by JS -->
      </div>
    </div>

    <!-- Selected pairs display -->
    <div class="selected-pairs-section">
      <div class="selected-pairs-title">Selected Pairs:</div>
      <div class="selected-pairs-chips" id="selected-dotplot-pairs-chips">
        <!-- Chips will be populated by JS -->
      </div>
    </div>

    <div style="display:flex;gap:12px;margin-top:16px">
      <button class="btn-primary" onclick="updateDotPlot()">Update Plot</button>
    </div>
  </div>

  <!-- Dot Plot -->
  <div class="plot-container">
    <div class="plot-title">
      Cross-Trace Per-Sample Comparison
      <span id="dotplot-count" style="color:var(--text-3);font-size:14px;font-weight:400;margin-left:8px"></span>
    </div>
    <div id="dot-plot"></div>
  </div>

  <!-- Selection Controls for Heatmap -->
  <div class="selection-section">
    <div class="selection-title">Select Pairs for Detailed Analysis</div>
    <p style="color:var(--text-2);font-size:13px;margin-bottom:16px">Click on pairs to select them for detailed subgoal × judge matrices below.</p>

    <!-- Search bar -->
    <div style="margin-bottom:16px">
      <input type="text" id="pair-search" class="search-input" placeholder="🔍 Search pairs (trace or sample)..." oninput="filterPairList(this.value)">
    </div>

    <!-- Action buttons -->
    <div style="display:flex;gap:12px;margin-bottom:16px">
      <button class="btn-secondary" onclick="selectAllPairs()">Select All</button>
      <button class="btn-secondary" onclick="clearAllPairs()">Clear All</button>
    </div>

    <!-- Pair list -->
    <div class="pair-list-container">
      <div class="pair-list" id="pair-list">
        <!-- List will be populated by JS -->
      </div>
    </div>

    <!-- Selected pairs display -->
    <div class="selected-pairs-section">
      <div class="selected-pairs-title">Selected Pairs:</div>
      <div class="selected-pairs-chips" id="selected-pairs-chips">
        <!-- Chips will be populated by JS -->
      </div>
    </div>

    <div style="display:flex;gap:12px;margin-top:16px">
      <button class="btn-primary" onclick="updateHeatmaps()">Update Heatmaps</button>
    </div>
  </div>

  <!-- Heatmap Comparison -->
  <div class="heatmap-section">
    <div class="plot-title">
      Binary Matrix: Subgoal × Judge for Selected Pairs
      <span id="heatmap-count" style="color:var(--text-3);font-size:14px;font-weight:400;margin-left:8px"></span>
    </div>
    <div id="heatmap-comparison" class="heatmap-comparison"></div>
  </div>
</div>

<script>
const DATA = {data_json};

// Toggle trace visibility in error summary table
function toggleTraces(cellId) {{
  const hiddenSpan = document.getElementById(`hidden_${{cellId}}`);
  const button = document.getElementById(`btn_${{cellId}}`);

  if (hiddenSpan.style.display === 'none') {{
    hiddenSpan.style.display = 'inline';
    button.textContent = 'Show less';
    button.style.marginLeft = '6px';
  }} else {{
    hiddenSpan.style.display = 'none';
    const hiddenText = hiddenSpan.textContent;
    const count = hiddenText.split(',').length;
    button.textContent = `+${{count}} more`;
  }}
}}

const COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#2ba7fc", "#ffb214", "#3ee03e", "#ff3738", "#cf90ff",
  "#c47869", "#ffa7ff", "#b2b2b2", "#ffff30", "#20ffff"
];

let selectedPairs = [];  // For heatmap pair selection
let selectedDotPlotPairs = [];  // For dot plot pair selection
const traceMap = {{}};
DATA.agent_runs.forEach((runId, i) => {{
  traceMap[runId] = `trace_${{i+1}}`;
}});

function escapeHtml(str) {{
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

// Initialize the pair selection list
function initPairList() {{
  const traces = DATA.agent_runs.map((runId, i) => `trace_${{i+1}}`);
  const samples = DATA.sample_ids;

  const listContainer = document.getElementById('pair-list');

  // Build list items grouped by sample
  samples.forEach((sample, sampleIdx) => {{
    traces.forEach((trace, traceIdx) => {{
      const item = document.createElement('div');
      item.className = 'pair-list-item';

      // Add separator class for last trace of each sample (except last sample)
      if (traceIdx === traces.length - 1 && sampleIdx < samples.length - 1) {{
        item.classList.add('sample-separator');
      }}

      item.textContent = `${{trace}} | ${{sample}}`;
      item.dataset.trace = trace;
      item.dataset.sample = sample;

      item.onclick = () => togglePairSelection(trace, sample, item);

      listContainer.appendChild(item);
    }});
  }});

  // Select first 4 pairs by default
  selectFirstFourPairs();
}}

function selectFirstFourPairs() {{
  const items = document.querySelectorAll('.pair-list-item');
  const limit = Math.min(4, items.length);

  for (let i = 0; i < limit; i++) {{
    const item = items[i];
    const trace = item.dataset.trace;
    const sample = item.dataset.sample;

    item.classList.add('selected');
    selectedPairs.push({{ trace, sample }});
  }}

  updateSelectedChips();
}}

function togglePairSelection(trace, sample, itemElement) {{
  const isSelected = itemElement.classList.contains('selected');

  if (isSelected) {{
    // Deselect
    itemElement.classList.remove('selected');
    selectedPairs = selectedPairs.filter(p => !(p.trace === trace && p.sample === sample));
  }} else {{
    // Select
    itemElement.classList.add('selected');
    selectedPairs.push({{ trace, sample }});
  }}

  updateSelectedChips();
}}

function updateSelectedChips() {{
  const container = document.getElementById('selected-pairs-chips');
  container.innerHTML = '';

  selectedPairs.forEach(pair => {{
    const chip = document.createElement('div');
    chip.className = 'pair-chip';

    const label = document.createElement('span');
    label.textContent = `${{pair.trace}} | ${{pair.sample}}`;
    chip.appendChild(label);

    const remove = document.createElement('span');
    remove.className = 'pair-chip-remove';
    remove.textContent = '×';
    remove.onclick = () => removePair(pair.trace, pair.sample);
    chip.appendChild(remove);

    container.appendChild(chip);
  }});
}}

function removePair(trace, sample) {{
  // Find and deselect the list item
  const items = document.querySelectorAll('.pair-list-item');
  items.forEach(item => {{
    if (item.dataset.trace === trace && item.dataset.sample === sample) {{
      item.classList.remove('selected');
    }}
  }});

  // Remove from selectedPairs
  selectedPairs = selectedPairs.filter(p => !(p.trace === trace && p.sample === sample));
  updateSelectedChips();
}}

function selectAllPairs() {{
  const items = document.querySelectorAll('.pair-list-item');
  selectedPairs = [];

  items.forEach(item => {{
    item.classList.add('selected');
    selectedPairs.push({{
      trace: item.dataset.trace,
      sample: item.dataset.sample
    }});
  }});

  updateSelectedChips();
}}

function clearAllPairs() {{
  const items = document.querySelectorAll('.pair-list-item');
  items.forEach(item => item.classList.remove('selected'));
  selectedPairs = [];
  updateSelectedChips();
}}

// Sample selector for dot plot

// Dot plot pair selector (unified sample + trace selector)
function initDotPlotPairList() {{
  const traces = DATA.agent_runs.map((runId, i) => `trace_${{i+1}}`);
  const samples = DATA.sample_ids;
  const listContainer = document.getElementById('dotplot-pair-list');

  // Build list grouped by sample with headers
  samples.forEach((sample, sampleIdx) => {{
    // Create sample header with inline "Select All" button
    const header = document.createElement('div');
    header.className = 'pair-list-sample-header';

    const sampleLabel = document.createElement('span');
    sampleLabel.textContent = sample;
    header.appendChild(sampleLabel);

    const selectAllBtn = document.createElement('button');
    selectAllBtn.className = 'pair-list-sample-header-btn';
    selectAllBtn.textContent = 'Select All ✓';
    selectAllBtn.onclick = (e) => {{
      e.stopPropagation();
      selectAllTracesForSample(sample);
    }};
    header.appendChild(selectAllBtn);

    listContainer.appendChild(header);

    // Create pair items for this sample
    traces.forEach((trace, traceIdx) => {{
      const item = document.createElement('div');
      item.className = 'pair-list-item indented';

      // Add separator after last trace of each sample (except the very last one)
      if (traceIdx === traces.length - 1 && sampleIdx < samples.length - 1) {{
        item.classList.add('sample-separator');
      }}

      item.textContent = `${{trace}} | ${{sample}}`;
      item.dataset.trace = trace;
      item.dataset.sample = sample;
      item.onclick = () => toggleDotPlotPairSelection(trace, sample, item);

      listContainer.appendChild(item);
    }});
  }});

  // Select all pairs by default (showing all data initially)
  selectAllDotPlotPairs();
}}

function toggleDotPlotPairSelection(trace, sample, itemElement) {{
  const isSelected = itemElement.classList.contains('selected');

  if (isSelected) {{
    // Deselect
    itemElement.classList.remove('selected');
    selectedDotPlotPairs = selectedDotPlotPairs.filter(p => !(p.trace === trace && p.sample === sample));
  }} else {{
    // Select
    itemElement.classList.add('selected');
    selectedDotPlotPairs.push({{ trace, sample }});
  }}

  updateSelectedDotPlotPairsChips();

  // CASCADE: Sync to detailed analysis selection
  syncDotPlotToDetailedAnalysis();
}}

function updateSelectedDotPlotPairsChips() {{
  const container = document.getElementById('selected-dotplot-pairs-chips');
  container.innerHTML = '';

  selectedDotPlotPairs.forEach(pair => {{
    const chip = document.createElement('div');
    chip.className = 'pair-chip';

    const label = document.createElement('span');
    label.textContent = `${{pair.trace}} | ${{pair.sample}}`;
    chip.appendChild(label);

    const remove = document.createElement('span');
    remove.className = 'pair-chip-remove';
    remove.textContent = '×';
    remove.onclick = () => removeDotPlotPair(pair.trace, pair.sample);
    chip.appendChild(remove);

    container.appendChild(chip);
  }});
}}

function removeDotPlotPair(trace, sample) {{
  // Find and deselect the list item
  const items = document.querySelectorAll('#dotplot-pair-list .pair-list-item');
  items.forEach(item => {{
    if (item.dataset.trace === trace && item.dataset.sample === sample) {{
      item.classList.remove('selected');
    }}
  }});

  // Remove from selectedDotPlotPairs
  selectedDotPlotPairs = selectedDotPlotPairs.filter(p => !(p.trace === trace && p.sample === sample));
  updateSelectedDotPlotPairsChips();

  // CASCADE: Sync to detailed analysis selection
  syncDotPlotToDetailedAnalysis();
}}

function selectAllTracesForSample(sample) {{
  const traces = DATA.agent_runs.map((runId, i) => `trace_${{i+1}}`);
  const items = document.querySelectorAll('#dotplot-pair-list .pair-list-item');

  traces.forEach(trace => {{
    // Find the item for this trace-sample pair
    items.forEach(item => {{
      if (item.dataset.trace === trace && item.dataset.sample === sample) {{
        if (!item.classList.contains('selected')) {{
          item.classList.add('selected');
          // Add to selectedDotPlotPairs if not already present
          const exists = selectedDotPlotPairs.some(p => p.trace === trace && p.sample === sample);
          if (!exists) {{
            selectedDotPlotPairs.push({{ trace, sample }});
          }}
        }}
      }}
    }});
  }});

  updateSelectedDotPlotPairsChips();

  // CASCADE: Sync to detailed analysis selection
  syncDotPlotToDetailedAnalysis();
}}

function selectAllDotPlotPairs() {{
  const items = document.querySelectorAll('#dotplot-pair-list .pair-list-item');
  selectedDotPlotPairs = [];

  items.forEach(item => {{
    if (item.dataset.trace && item.dataset.sample) {{
      item.classList.add('selected');
      selectedDotPlotPairs.push({{
        trace: item.dataset.trace,
        sample: item.dataset.sample
      }});
    }}
  }});

  updateSelectedDotPlotPairsChips();

  // CASCADE: Sync to detailed analysis selection
  syncDotPlotToDetailedAnalysis();
}}

function clearAllDotPlotPairs() {{
  const items = document.querySelectorAll('#dotplot-pair-list .pair-list-item');
  items.forEach(item => item.classList.remove('selected'));
  selectedDotPlotPairs = [];
  updateSelectedDotPlotPairsChips();

  // CASCADE: Sync to detailed analysis selection
  syncDotPlotToDetailedAnalysis();
}}

// CASCADE: Sync dot plot selections to detailed analysis
function syncDotPlotToDetailedAnalysis() {{
  // Clear detailed analysis selections
  selectedPairs = [];

  // Remove all selected classes from detailed analysis items
  const detailedItems = document.querySelectorAll('#pair-list .pair-list-item');
  detailedItems.forEach(item => item.classList.remove('selected'));

  // Copy selections from dot plot to detailed analysis
  selectedDotPlotPairs.forEach(dotPlotPair => {{
    // Add to selectedPairs array
    selectedPairs.push({{
      trace: dotPlotPair.trace,
      sample: dotPlotPair.sample
    }});

    // Find and select the corresponding item in detailed analysis list
    detailedItems.forEach(item => {{
      if (item.dataset.trace === dotPlotPair.trace && item.dataset.sample === dotPlotPair.sample) {{
        item.classList.add('selected');
      }}
    }});
  }});

  // Update the chips display for detailed analysis
  updateSelectedChips();
}}

function updateDotPlot() {{
  renderDotPlot();

  // Scroll to plot for visual feedback
  const plotContainer = document.querySelector('.plot-container');
  if (plotContainer) {{
    plotContainer.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
  }}
}}

// Search/filter functions
function filterPairList(searchText) {{
  const items = document.querySelectorAll('#pair-list .pair-list-item');
  const search = searchText.toLowerCase().trim();

  items.forEach(item => {{
    const text = item.textContent.toLowerCase();
    if (text.includes(search)) {{
      item.classList.remove('hidden');
    }} else {{
      item.classList.add('hidden');
    }}
  }});
}}

function filterDotPlotPairList(searchText) {{
  const items = document.querySelectorAll('#dotplot-pair-list .pair-list-item');
  const headers = document.querySelectorAll('#dotplot-pair-list .pair-list-sample-header');
  const search = searchText.toLowerCase().trim();

  // Filter pair items
  items.forEach(item => {{
    const text = item.textContent.toLowerCase();
    if (text.includes(search)) {{
      item.classList.remove('hidden');
    }} else {{
      item.classList.add('hidden');
    }}
  }});

  // Also filter headers - hide if no visible items for that sample
  headers.forEach(header => {{
    const sampleName = header.querySelector('span').textContent;
    const sampleItems = Array.from(items).filter(item =>
      item.dataset.sample === sampleName
    );
    const hasVisibleItems = sampleItems.some(item => !item.classList.contains('hidden'));

    if (search === '' || hasVisibleItems) {{
      header.classList.remove('hidden');
    }} else {{
      header.classList.add('hidden');
    }}
  }});
}}

function updateHeatmaps() {{
  renderHeatmaps();

  // Scroll to heatmap section for visual feedback
  const heatmapSection = document.querySelector('.heatmap-section');
  if (heatmapSection) {{
    heatmapSection.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
  }}
}}

function renderDotPlot() {{
  // Use selectedDotPlotPairs if any, otherwise show all data
  let samplesInPlot, tracesInPlot;

  if (selectedDotPlotPairs.length > 0) {{
    // Extract unique samples and traces from selected pairs
    const samplesSet = new Set(selectedDotPlotPairs.map(p => p.sample));
    const tracesSet = new Set(selectedDotPlotPairs.map(p => p.trace));
    samplesInPlot = Array.from(samplesSet);
    tracesInPlot = Array.from(tracesSet);

    // Sort to maintain consistent ordering
    samplesInPlot.sort();
    tracesInPlot.sort((a, b) => {{
      const aNum = parseInt(a.split('_')[1]);
      const bNum = parseInt(b.split('_')[1]);
      return aNum - bNum;
    }});
  }} else {{
    // Default: show all
    samplesInPlot = DATA.sample_ids;
    tracesInPlot = DATA.agent_runs.map((runId, i) => `trace_${{i+1}}`);
  }}

  // Update count display
  const countSpan = document.getElementById('dotplot-count');
  if (countSpan) {{
    countSpan.textContent = `(showing ${{tracesInPlot.length}} trace${{tracesInPlot.length !== 1 ? 's' : ''}} × ${{samplesInPlot.length}} sample${{samplesInPlot.length !== 1 ? 's' : ''}})`;
  }}

  const sampleToX = {{}};
  samplesInPlot.forEach((s, i) => sampleToX[s] = i);

  const numTraces = tracesInPlot.length;
  const totalWidth = 0.7;
  const dotWidth = totalWidth / numTraces;

  // Build plot traces - only show selected pairs
  const plotTraces = [];
  const traceShownInLegend = new Set();

  DATA.mean_data.forEach(meanRow => {{
    const trace = traceMap[meanRow.agent_run_id];
    const sample = meanRow.sample_idx;

    // Check if this specific pair is selected (or if showing all)
    if (selectedDotPlotPairs.length > 0) {{
      const isSelected = selectedDotPlotPairs.some(p => p.trace === trace && p.sample === sample);
      if (!isSelected) return;
    }} else {{
      // Default: include all
      if (!samplesInPlot.includes(sample)) return;
      if (!tracesInPlot.includes(trace)) return;
    }}

    const traceIdx = tracesInPlot.indexOf(trace);
    const sampleX = sampleToX[sample];
    const x = sampleX - (totalWidth / 2) + (dotWidth / 2) + traceIdx * dotWidth;

    const colorIdx = parseInt(trace.split('_')[1]) - 1;
    const color = COLORS[colorIdx % COLORS.length];

    const showLegend = !traceShownInLegend.has(trace);
    if (showLegend) traceShownInLegend.add(trace);

    plotTraces.push({{
      x: [x],
      y: [meanRow.SB_token_prob_mean],
      error_y: {{
        type: 'data',
        array: [meanRow.SB_token_prob_SD_aggregated],
        visible: true,
        thickness: 2,
        width: 4
      }},
      mode: 'markers',
      marker: {{ size: 14, color: color, line: {{ width: 1, color: 'black' }} }},
      name: trace,
      legendgroup: trace,
      showlegend: showLegend,
      hovertemplate: `Trace: ${{trace}}<br>Sample: ${{sample}}<br>Mean: %{{y:.3f}}<extra></extra>`
    }});
  }});

  // Layout with vertical bands
  const shapes = samplesInPlot.map((s, i) => ({{
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: i - 0.5,
    x1: i + 0.5,
    y0: 0,
    y1: 1,
    fillcolor: i % 2 === 0 ? '#f0f0f0' : '#ffffff',
    opacity: 0.25,
    layer: 'below',
    line: {{ width: 0 }}
  }}));

  const layout = {{
    xaxis: {{
      tickmode: 'array',
      tickvals: samplesInPlot.map((s, i) => i),
      ticktext: samplesInPlot.map(s => `${{s}}`),
      title: {{
        text: 'Sample Index',
        standoff: 40
      }},
      showgrid: false,
      zeroline: false
    }},
    yaxis: {{
      title: 'E[progress(i, Gi, τi)]',
      range: [0, 1.15],
      showgrid: true,
      zeroline: true
    }},
    legend: {{ title: {{ text: 'Trace' }} }},
    font: {{ family: 'Georgia', size: 14 }},
    shapes: shapes,
    margin: {{ l: 60, r: 40, t: 40, b: 120 }},
    plot_bgcolor: 'white',
    paper_bgcolor: 'white'
  }};

  Plotly.newPlot('dot-plot', plotTraces, layout, {{ responsive: true }});
}}

function renderHeatmaps() {{
  const container = document.getElementById('heatmap-comparison');
  const countSpan = document.getElementById('heatmap-count');

  if (selectedPairs.length === 0) {{
    container.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-3)">Select samples or pairs to view heatmaps</div>';
    countSpan.textContent = '';
    return;
  }}

  // Update count display
  countSpan.textContent = `(showing ${{selectedPairs.length}} pair${{selectedPairs.length !== 1 ? 's' : ''}})`;

  // Show all selected pairs (not limited to 4)
  const displayPairs = selectedPairs;

  let html = '';
  displayPairs.forEach((pair, pairIdx) => {{
    const runId = Object.keys(traceMap).find(k => traceMap[k] === pair.trace);

    // Get subgoals for this pair
    const subgoalRows = DATA.full_data.filter(d =>
      d.agent_run_id === runId && d.sample_idx === pair.sample
    ).sort((a, b) => a.subgoal_idx - b.subgoal_idx);

    if (subgoalRows.length === 0) {{
      html += `<div class="heatmap-panel"><div class="heatmap-panel-title">${{pair.trace}} | ${{pair.sample}}</div><p style="color:var(--text-3);text-align:center">No data</p></div>`;
      return;
    }}

    html += `<div class="heatmap-panel" style="--judge-count:${{DATA.num_judges}}">`;
    html += `<div class="heatmap-panel-title">${{pair.trace}} | ${{pair.sample}}</div>`;

    // Build heatmap grid
    html += '<div class="heatmap-grid">';

    // Header row
    html += '<div class="heatmap-row"><div class="heatmap-label"></div>';
    for (let j = 0; j < DATA.num_judges; j++) {{
      html += `<div class="heatmap-header-cell">Judge ${{j + 1}}</div>`;
    }}
    html += '</div>';

    // Data rows
    subgoalRows.forEach(row => {{
      html += `<div class="heatmap-row">`;
      html += `<div class="heatmap-label" title="${{escapeHtml(row.subgoal_detail)}}">Subgoal ${{row.subgoal_idx}}</div>`;

      for (let j = 0; j < DATA.num_judges; j++) {{
        const score = row[`pred_${{j}}_score`];
        const cls = score === 1.0 ? 'cell-complete' : 'cell-incomplete';
        const val = score === 1.0 ? '1' : '0';
        html += `<div class="heatmap-cell ${{cls}}" title="Subgoal ${{row.subgoal_idx}}, Judge ${{j + 1}}: ${{val}}">${{val}}</div>`;
      }}

      html += '</div>';
    }});

    html += '</div>';  // heatmap-grid

    // Legend
    html += `
      <div class="legend">
        <div class="legend-item"><div class="legend-swatch cell-complete"></div> Complete (1)</div>
        <div class="legend-item"><div class="legend-swatch cell-incomplete"></div> Incomplete (0)</div>
      </div>
    `;

    // Judge details
    html += `<div class="judge-details-container" id="judge-details-${{pairIdx}}">`;
    html += `<div class="judge-details-title">Judge Details</div>`;

    // Subgoal selector
    html += `<div class="subgoal-selector">
      <select onchange="updateJudgeDetails(${{pairIdx}}, this.value, '${{runId}}', '${{pair.sample}}')">`;
    subgoalRows.forEach(row => {{
      html += `<option value="${{row.subgoal_idx}}">Subgoal ${{row.subgoal_idx}}: ${{row.subgoal_detail.substring(0, 60)}}...</option>`;
    }});
    html += `</select></div>`;

    // Initial judge details for first subgoal
    const firstSubgoal = subgoalRows[0];
    html += buildJudgeDetails(firstSubgoal, pairIdx);

    html += '</div>';  // judge-details-container
    html += '</div>';  // heatmap-panel
  }});

  container.innerHTML = html;
}}

function buildJudgeDetails(subgoalRow, pairIdx) {{
  let html = '<div id="judge-details-content-' + pairIdx + '">';

  for (let j = 0; j < DATA.num_judges; j++) {{
    const pred = subgoalRow[`pred_${{j}}`] || 'N/A';
    const judgeInput = subgoalRow.judge_model_input || 'N/A';

    html += `
      <div class="judge-item">
        <div class="judge-header" onclick="toggleJudge(this)">
          Judge ${{j + 1}}
          <span>▼</span>
        </div>
        <div class="judge-content">
          <div class="judge-field">
            <div class="judge-field-label">Subgoal</div>
            <div class="judge-field-value">${{escapeHtml(subgoalRow.subgoal_detail)}}</div>
          </div>
          <div class="judge-field">
            <div class="judge-field-label">Judge Explanation</div>
            <div class="judge-field-value">${{escapeHtml(pred)}}</div>
          </div>
          <div class="judge-field">
            <div class="judge-field-label">Judge Input</div>
            <div class="judge-field-code">${{escapeHtml(judgeInput)}}</div>
          </div>
        </div>
      </div>
    `;
  }}

  html += '</div>';
  return html;
}}

function toggleJudge(header) {{
  const content = header.nextElementSibling;
  const arrow = header.querySelector('span');
  if (content.classList.contains('open')) {{
    content.classList.remove('open');
    arrow.textContent = '▼';
  }} else {{
    content.classList.add('open');
    arrow.textContent = '▲';
  }}
}}

function updateJudgeDetails(pairIdx, subgoalIdx, runId, sampleId) {{
  const subgoalRow = DATA.full_data.find(d =>
    d.agent_run_id === runId &&
    d.sample_idx === sampleId &&
    d.subgoal_idx === parseInt(subgoalIdx)
  );

  if (subgoalRow) {{
    const detailsHtml = buildJudgeDetails(subgoalRow, pairIdx);
    document.getElementById('judge-details-content-' + pairIdx).outerHTML = detailsHtml;
  }}
}}

// Initialize - render dot plot immediately and initialize selectors
window.onload = function() {{
  // Initialize the unified dot plot pair selector
  initDotPlotPairList();

  // Render dot plot with all pairs selected (default)
  renderDotPlot();

  // Initialize the pair selection list for heatmap
  initPairList();

  // Render initial heatmaps
  renderHeatmaps();
}};
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main Generator Functions
# ---------------------------------------------------------------------------


def generate_folder_name(entry: Dict[str, Any], dataset_name: str) -> str:
    """Generate a readable folder name: dataset_model_persona_id"""
    model = entry.get("agent_model", "").split("/")[-1].replace("-", "_").replace(".", "_")
    persona = entry.get("user_proxy_persona", "unknown")
    entry_id = entry["id"][:8]
    # Normalize dataset name: remove special chars, convert to lowercase, replace spaces/hyphens with underscores
    dataset = dataset_name.lower()
    # Remove special characters like em-dash, en-dash, etc
    dataset = dataset.replace("–", " ").replace("—", " ").replace("-", " ")
    # Replace multiple spaces with single space, then convert spaces to underscores
    dataset = " ".join(dataset.split()).replace(" ", "_")
    return f"{dataset}_{model}_{persona}_{entry_id}"


def generate_error_analysis_pages(
    entry: Dict[str, Any],
    dataset_name: str,
    source_folder: Path,
) -> Optional[Path]:
    """
    Generate error analysis HTML page for a leaderboard entry.

    Args:
        entry: Entry dictionary with model info
        dataset_name: Name of the dataset
        source_folder: Path to the experiment output folder

    Returns:
        Path to the generated index.html, or None if no error analysis data found
    """
    # Look for error_analysis.pkl in source folder
    pkl_path = source_folder / "error_analysis.pkl"

    if not pkl_path.exists():
        print(f"  [skip] No error_analysis.pkl found in {source_folder.name}")
        return None

    try:
        # Load and extract data
        print(f"  Loading error analysis data from {pkl_path.name}...")
        data_samples, error_result = load_error_analysis_data(pkl_path)
        error_data = extract_error_data(data_samples, error_result)

        # Generate folder name
        folder_name = generate_folder_name(entry, dataset_name)

        # Create output directory
        out_dir = ERROR_ANALYSIS_BASE / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate HTML page
        html = build_error_analysis_page(entry, dataset_name, error_data, folder_name)

        # Write to file
        output_path = out_dir / "index.html"
        output_path.write_text(html, encoding="utf-8")

        print(f"  Error analysis → {output_path}")
        return output_path

    except Exception as e:
        print(f"  [error] Failed to generate error analysis: {e}")
        import traceback

        traceback.print_exc()
        return None
