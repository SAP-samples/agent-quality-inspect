import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional

st.set_page_config(layout="wide")

@st.cache_data
def load_token_prob_bar_data(csv_path):
    """
    Legacy function to load data from CSV for backward compatibility.
    Note: CSV files may use 'filename' column name for backward compatibility,
    but it will be renamed to 'agent_run_id' internally.
    """
    df = pd.read_csv(csv_path)
    
    # Support both 'filename' (legacy) and 'agent_run_id' (new) column names
    if 'filename' in df.columns and 'agent_run_id' not in df.columns:
        df = df.rename(columns={'filename': 'agent_run_id'})
    
    score_cols = [col for col in df.columns if col.startswith("pred_") and col.endswith("_score")]
    df["SB_token_prob"] = df[score_cols].sum(axis=1) / len(score_cols)
    mean_df = df.groupby(["agent_run_id", "sample_idx"])["SB_token_prob"].mean().reset_index()
    mean_df = mean_df.rename(columns={"SB_token_prob": "SB_token_prob_mean"})
    def aggregate_sd(pi):
        n = len(pi)
        pi = pi.to_numpy()
        std = np.sqrt(np.sum(pi * (1 - pi)) / (n**2))
        return std
    agg_sd = df.groupby(["agent_run_id", "sample_idx"])["SB_token_prob"].apply(aggregate_sd).reset_index()
    agg_sd = agg_sd.rename(columns={"SB_token_prob": "SB_token_prob_SD_aggregated"})
    mean_df = mean_df.merge(agg_sd, on=["agent_run_id", "sample_idx"], how="left")
    return mean_df, df

def trace_sort_key(trace_name):
    return int(trace_name.split('_')[1])

def extract_subgoal_from_explanation(explanation):
    if pd.isna(explanation):
        return ""
    return explanation.split('\n')[0].strip()

def error_summary_matrix_tab_simple(full_df, agent_run_id_to_trace, traces, samples, selected_pairs_callback):
    # Only proceed if cluster_label and final_error_type columns exist
    if "cluster_label" not in full_df.columns or "final_error_type" not in full_df.columns:
        return

    # Use st.toggle for hide/unhide functionality (Streamlit >=1.32)
    show_table = st.toggle(
        "Show Error Category Summary Table",
        value=st.session_state.get("show_error_summary_table", True),
        key="toggle_error_summary_table"
    )
    st.session_state["show_error_summary_table"] = show_table

    if not show_table:
        return

    error_df = full_df.dropna(subset=["cluster_label"])
    error_df = error_df[error_df["cluster_label"] != ""]
    # Omit rows where cluster_label contains BOTH "no" AND "error" (case-insensitive, anywhere in the string)
    error_df = error_df[~(
        error_df["cluster_label"].str.lower().str.contains("no") &
        error_df["cluster_label"].str.lower().str.contains("error")
    )]
    if error_df.empty:
        return

    # Prepare mapping: (trace, sample_idx) -> T{n}
    sample_display_map = {}
    for trace in traces:
        for idx, s in enumerate(samples):
            t_label = f"T{idx+1}" if not isinstance(s, str) else s
            sample_display_map[(trace, s)] = t_label

    # Show all columns of samples
    sample_cols = samples
    # Table columns: use sample_idx as column names
    col_names = [str(s) for s in sample_cols] + ["Total Cases"]

    # Build error matrix: cluster_label × sample
    cluster_labels = sorted(error_df["cluster_label"].unique())
    matrix_data = []
    for row_idx, cluster in enumerate(cluster_labels):
        row = []
        cluster_cases = 0
        for col_idx, s in enumerate(sample_cols):
            tns = []
            for trace in traces:
                agent_run_id = [k for k, v in agent_run_id_to_trace.items() if v == trace][0]
                mask = (
                    (error_df["agent_run_id"] == agent_run_id) &
                    (error_df["sample_idx"] == s) &
                    (error_df["cluster_label"] == cluster)
                )
                idxs = error_df[mask].index.tolist()
                if idxs:
                    trace_idx = trace if trace.startswith("trace_") else f"trace_{trace}"
                    tns.append(trace_idx)
                    cluster_cases += len(idxs)
            if tns:
                row.append(", ".join(tns))
            else:
                row.append("–")
        row.append(cluster_cases)
        matrix_data.append(row)

    # Set column names, with first column as "LLM summarized error category"
    # Format sample column names with "sample_" prefix
    sample_col_names = [f"sample_{s}" if not isinstance(s, str) or not s.startswith("sample_") else s for s in sample_cols]
    col_names = ["LLM summarized error category"] + sample_col_names + ["Total Cases"]

    # Prepare DataFrame with cluster_labels as first column (bold)
    matrix_df = pd.DataFrame(matrix_data, columns=col_names[1:])
    matrix_df.insert(0, col_names[0], cluster_labels)

    # Use st.markdown for title and style
    st.markdown("""
        <h3 style="font-size:1.5em;">Error Category Summary (Columns: sample_idx, Content: trace idx)</h3>
        <style>
        .big-table td, .big-table th {font-size: 1.3em !important;}
        </style>
    """, unsafe_allow_html=True)

    # Use st.dataframe for full width and st.write for bold first column
    def style_table(df):
        return df.style.set_properties(**{'font-size': '20px'}).set_properties(subset=[col_names[0]], **{'font-weight': 'bold'})

    st.dataframe(style_table(matrix_df), use_container_width=True)
    st.markdown(
        "<span style='font-size:1.1em; color:#b71c1c;'>"
        "The whole summary is generated and clustered by LLM, only used for reference and it may not be 100% correct. "
        "Please refer to the below details for double confirm."
        "</span>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

def consistency_viz_ui(csv_path: Optional[str] = None, mean_df: Optional[pd.DataFrame] = None, full_df: Optional[pd.DataFrame] = None):
    """
    Main UI function for consistency visualization.
    
    Args:
        csv_path: Path to CSV file (for backward compatibility)
        mean_df: Pre-computed mean DataFrame with columns: agent_run_id, sample_idx, SB_token_prob_mean, SB_token_prob_SD_aggregated
        full_df: Pre-computed full DataFrame with detailed subgoal information
    """
    st.markdown("""
        <h2 style="font-family: 'Georgia', serif; font-weight: 700; color: #1a237e; margin-bottom: 0.5em;">
            Cross-Trace, Per-Sample Pairwise Comparison
        </h2>
        <hr style="margin-top:0;margin-bottom:1.5em;">
    """, unsafe_allow_html=True)

    # Load data from either CSV or in-memory DataFrames
    if mean_df is None or full_df is None:
        if csv_path is None:
            st.error("Either csv_path or both mean_df and full_df must be provided.")
            return
        mean_df, full_df = load_token_prob_bar_data(csv_path)
    
    # Ensure mean_df and full_df are not empty
    if mean_df.empty:
        st.warning("No data available to display.")
        return
    if full_df.empty:
        st.warning("No data available to display. Ensure that the custom runner's result has at least one successful evaluation run.")
        return
    agent_run_id_list = sorted(mean_df["agent_run_id"].unique())
    agent_run_id_to_trace = {run_id: f"trace_{i+1}" for i, run_id in enumerate(agent_run_id_list)}
    mean_df["trace"] = mean_df["agent_run_id"].map(agent_run_id_to_trace)
    full_df["trace"] = full_df["agent_run_id"].map(agent_run_id_to_trace)

    traces = sorted(mean_df["trace"].unique(), key=trace_sort_key)
    samples = sorted(mean_df["sample_idx"].unique())
    num_traces = len(traces)

    # --- Show error summary matrix above the plot ---
    def select_trace_sample(trace, sample_idx):
        st.session_state["selected_trace"] = trace
        st.session_state["selected_sample_idx"] = sample_idx
        st.experimental_rerun()

    error_summary_matrix_tab_simple(full_df, agent_run_id_to_trace, traces, samples, select_trace_sample)

    # --- Add selection widget for dots ---
    # Define variables needed for plotting
    colors =  [
    "#1f77b4", "#2ba7fc",
    "#ff7f0e", "#ffb214",
    "#2ca02c", "#3ee03e",
    "#d62728", "#ff3738",
    "#9467bd", "#cf90ff",
    "#8c564b", "#c47869",
    "#e377c2", "#ffa7ff",
    "#7f7f7f", "#b2b2b2",
    "#bcbd22", "#ffff30",
    "#17becf", "#20ffff"
]
    total_width = 0.7
    dot_width = total_width / num_traces

    # Build all possible trace/sample pairs
    all_dot_options = []
    dot_label_to_pair = {}
    for trace in traces:
        for idx, sample_idx in enumerate(samples):
            if isinstance(sample_idx, str):
                label = f"{trace} | {sample_idx}"
            else:
                label = f"{trace} | Sample {sample_idx}"
            all_dot_options.append(label)
            dot_label_to_pair[label] = (trace, sample_idx)

    st.markdown("**Select which dots to show on the plot:**")
    selection_mode = st.radio(
        "Dot selection mode",
        ["Custom selection", "All traces for multiple samples"],
        horizontal=True,
        key="dot_selection_mode"
    )

    if selection_mode == "All traces for multiple samples":
        def _sample_label(s):
            return s if isinstance(s, str) else f"Sample {s}"
        chosen_samples = st.multiselect(
            "Select samples to display all traces",
            samples,
            default=samples[: min(3, len(samples))],
            format_func=_sample_label,
            key="all_traces_multi_sample_selector"
        )
        chosen_samples_sorted = sorted(chosen_samples, key=lambda s: (str(s) if isinstance(s, str) else s))
        selected_dot_pairs = []
        selected_dot_labels = []
        for cs in chosen_samples_sorted:
            for trace in traces:
                selected_dot_pairs.append((trace, cs))
                label = cs if isinstance(cs, str) else f"Sample {cs}"
                selected_dot_labels.append(f"{trace} | {label}")
    else:  # Custom selection
        selected_dot_labels = st.multiselect(
            "Choose dots (trace/sample pairs)",
            all_dot_options,
            default=all_dot_options,  # default: show all initially
            help="Select which trace/sample pairs to show as dots on the plot.",
            key="custom_pairs_multiselect"
        )
        selected_dot_pairs = [dot_label_to_pair[label] for label in selected_dot_labels]

    # Prepare data for selected dots only
    PR_means_dot_plot = []
    PR_CI_dot_plot = []
    dot_x = []
    dot_trace_names = []
    dot_sample_idxs = []
    color_map = {trace: colors[i % len(colors)] for i, trace in enumerate(traces)}
    # Dynamically determine x-axis ticks and labels based on selected samples
    selected_samples = sorted(set([sample_idx for _, sample_idx in selected_dot_pairs]), key=lambda s: (str(s) if isinstance(s, str) else s))
    selected_sample_to_x = {s: i for i, s in enumerate(selected_samples)}
    for i, (trace, sample_idx) in enumerate(selected_dot_pairs):
        # Find the corresponding mean and SD
        subdf = mean_df[(mean_df["trace"] == trace) & (mean_df["sample_idx"] == sample_idx)]
        if not subdf.empty:
            mean = subdf["SB_token_prob_mean"].values[0]
            ci = subdf["SB_token_prob_SD_aggregated"].values[0]
            # Calculate x position for this dot (now based on selected samples only)
            trace_idx = traces.index(trace)
            sample_idx_pos = selected_sample_to_x[sample_idx]
            x_val = sample_idx_pos - (total_width / 2) + (dot_width / 2) + trace_idx * dot_width
            dot_x.append(x_val)
            PR_means_dot_plot.append(mean)
            PR_CI_dot_plot.append(ci)
            dot_trace_names.append(trace)
            dot_sample_idxs.append(sample_idx)

    # Provide CSV download for currently displayed dot plot data
    if PR_means_dot_plot:
        export_rows = []
        for idx_plot in range(len(PR_means_dot_plot)):
            trace_name = dot_trace_names[idx_plot]
            sample_idx_val = dot_sample_idxs[idx_plot]
            mean_val = PR_means_dot_plot[idx_plot]
            sd_val = PR_CI_dot_plot[idx_plot]
            agent_run_id = next((run_id for run_id, tr in agent_run_id_to_trace.items() if tr == trace_name), None)
            export_rows.append({
                "trace": trace_name,
                "agent_run_id": agent_run_id,
                "sample_idx": sample_idx_val,
                "SB_token_prob_mean": mean_val,
                "SB_token_prob_SD_aggregated": sd_val
            })
        dot_export_df = pd.DataFrame(export_rows)
        col_dl, col_preview = st.columns([1,3])
        with col_dl:
            st.download_button(
                label="Download dot plot data (CSV)",
                data=dot_export_df.to_csv(index=False).encode('utf-8'),
                file_name='dot_plot_data.csv',
                mime='text/csv',
                help="Exports mean and aggregated SD for each displayed trace/sample pair."
            )
        with col_preview:
            st.dataframe(dot_export_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    # Plot each dot individually, but group by trace to avoid duplicate legend entries
    displayed_pairs_for_plot = []  # (trace_name, sample_idx) in plotting order
    trace_shown_in_legend = set()  # Track which traces have been shown in legend
    
    for i in range(len(dot_x)):
        displayed_pairs_for_plot.append((dot_trace_names[i], dot_sample_idxs[i]))
        trace_name = dot_trace_names[i]
        
        # Only show the first occurrence of each trace in the legend
        show_legend = trace_name not in trace_shown_in_legend
        if show_legend:
            trace_shown_in_legend.add(trace_name)
        
        fig.add_trace(go.Scatter(
            x=[dot_x[i]],
            y=[PR_means_dot_plot[i]],
            error_y=dict(type='data', array=[PR_CI_dot_plot[i]], visible=True, thickness=2, width=4),
            mode='markers',
            marker=dict(size=14, color=color_map[trace_name], line=dict(width=1, color='black')),
            name=trace_name,
            legendgroup=trace_name,  # Group all dots of the same trace
            showlegend=show_legend,  # Only show first occurrence in legend
            customdata=[[trace_name, dot_sample_idxs[i]]],
            hovertemplate="Trace: %{customdata[0]}<br>Sample: %{customdata[1]}<br>Mean: %{y:.3f}<extra></extra>"
        ))

    # Dynamically add vrects for selected samples only
    for i in range(len(selected_samples)):
        fig.add_vrect(
            x0=i-0.5, x1=i+0.5,
            fillcolor="#f0f0f0" if i % 2 == 0 else "#ffffff",
            opacity=0.25, layer="below", line_width=0,
        )

    # Dynamically set x-axis ticks and labels
    # Detect dark mode and set background colors accordingly
    bg_color = st.get_option("theme.backgroundColor")
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[selected_sample_to_x[s] for s in selected_samples],
            ticktext=[
                s if (isinstance(s, str))
                else f"sample {s}" for s in selected_samples
            ],
            title="Sample Index"
        ),
        yaxis=dict(title="E[progress(i, Gi, τi)]", range=[0, 1.15]),
        legend_title="Trace",
        font=dict(family="Georgia", size=16),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title=dict(
            text="<b>Estimating Progress Rate and its Variance</b>",
            font=dict(size=18, color="#7882f1"),
            x=0.5
        ),
        width=1200,
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True, key="dotplot", select_data=True)
    select_data = st.session_state.get("plotly_chart-dotplot-selected-data", None)
    selected_pairs = []
    # If mode is all traces one sample, pre-populate selected pairs so downstream heatmap is auto-populated
    if selection_mode == "All traces for multiple samples":
        # All plotted dots correspond to chosen samples/traces, so auto-select them for matrix
        selected_pairs = displayed_pairs_for_plot.copy()
    elif select_data and "points" in select_data:
        for pt in select_data["points"]:
            curve_num = pt.get("curveNumber")
            if curve_num is not None and 0 <= curve_num < len(displayed_pairs_for_plot):
                selected_pairs.append(displayed_pairs_for_plot[curve_num])

    # --- Fallback: dropdown selection if dot select not working ---
    if selection_mode == "All traces for multiple samples":
        selected_pairs = []

    st.markdown("Select trace/sample pairs manually below:")
    trace_options = []
    for s in samples:
        for trace in traces:
            if isinstance(s, str):
                label = f"{trace} | {s}"
            else:
                label = f"{trace} | Sample {s}"
            trace_options.append(label)
    default_selection = []
    for (trace, s) in selected_pairs:
        if isinstance(s, str):
            default_selection.append(f"{trace} | {s}")
        else:
            default_selection.append(f"{trace} | Sample {s}")
    manual_selection = st.multiselect(
        "Select Trace-Sample Pairs",
        trace_options,
        default=default_selection,
        help="Use this dropdown to select trace/sample pairs.",
        key="manual_selection_multiselect"
    )
    # Rebuild selected_pairs strictly from manual selection to avoid stale items
    rebuilt_pairs = []
    for item in manual_selection:
        if " | Sample " in item:
            trace, _, sample_part = item.partition(" | Sample ")
            sample_idx = int(sample_part)
        else:
            trace, _, sample_part = item.partition(" | ")
            sample_idx = sample_part  # string sample id
        rebuilt_pairs.append((trace, sample_idx))
    selected_pairs = rebuilt_pairs

    if not selected_pairs:
        st.info("Use the dropdown to show the matrix comparison.")
        return

    # --- Heatmap for selected pairs ---
    st.markdown("### Binary Matrix: Subgoal × Judge for Selected Traces/Samples")

    # Collect all subgoals for selected pairs
    dfs = []
    for trace, sample_idx in selected_pairs:
        agent_run_id = [k for k, v in agent_run_id_to_trace.items() if v == trace][0]
        df_sel = full_df[(full_df["agent_run_id"] == agent_run_id) & (full_df["sample_idx"] == sample_idx)]
        if not df_sel.empty:
            df_sel = df_sel.copy()
            df_sel["trace"] = trace
            dfs.append(df_sel)
    if not dfs:
        st.warning("No data found for selected points.")
        return
    compare_df = pd.concat(dfs, ignore_index=True)
    subgoals = compare_df["subgoal_idx"].unique() if "subgoal_idx" in compare_df.columns else []

    judge_cols = [col for col in compare_df.columns if col.startswith("pred_") and col.endswith("_score")]
    num_judges = len(judge_cols)
    judges = [f"Judge {i+1}" for i in range(num_judges)]

    # Subgoal selection and heatmap order toggle
    st.markdown("#### Options")
    col_order, col_show = st.columns([1, 2])
    with col_order:
        reverse_order = st.checkbox("Reverse subgoal order in matrix", value=False)
    with col_show:
        show_all_subgoals = st.checkbox("Show all subgoals in matrix", value=True)

    # Subgoal selection
    if len(subgoals) > 0 and not show_all_subgoals:
        selected_subgoal = st.selectbox("Select Subgoal", subgoals)
    else:
        selected_subgoal = None

    # --- Horizontal comparison: show all selected pairs side by side ---
    cols = st.columns(len(selected_pairs))
    for idx, (trace, sample_idx) in enumerate(selected_pairs):
        with cols[idx]:
            # Display sample label based on type
            if isinstance(sample_idx, str):
                sample_label = sample_idx
            else:
                sample_label = f"Sample {sample_idx}"
            st.markdown(f"#### {trace} | {sample_label}")
            agent_run_id = [k for k, v in agent_run_id_to_trace.items() if v == trace][0]
            df_pair = full_df[(full_df["agent_run_id"] == agent_run_id) & (full_df["sample_idx"] == sample_idx)]
            if df_pair.empty:
                st.write("No subgoal data.")
                continue

            judge_cols = [f"pred_{i}_score" for i in range(num_judges)]
            # Prepare heatmap data
            if show_all_subgoals:
                heatmap_data = df_pair[judge_cols].values
                subgoal_labels = df_pair["subgoal_idx"].tolist() if "subgoal_idx" in df_pair.columns else [0]
                if reverse_order:
                    heatmap_data = heatmap_data[::-1]
                    subgoal_labels = subgoal_labels[::-1]
            else:
                # Only show selected subgoal
                if selected_subgoal and "subgoal_idx" in df_pair.columns:
                    df_pair = df_pair[df_pair["subgoal_idx"] == selected_subgoal]
                heatmap_data = df_pair[judge_cols].values
                subgoal_labels = df_pair["subgoal_idx"].tolist() if "subgoal_idx" in df_pair.columns else [0]

            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=subgoal_labels,
                columns=judges
            )
            # Use a categorical color scale for binary data
            binary_colorscale = [[0, "#e0e0e0"], [1, "#1976d2"]]  # 0: gray, 1: blue

            fig = go.Figure(
                data=go.Heatmap(
                    z=heatmap_df.values,
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    colorscale=binary_colorscale,
                    colorbar=None,  # <-- Remove the colorbar legend
                    hovertemplate="Subgoal: %{y}<br>Judge: %{x}<br>Value: %{z}<extra></extra>",
                    showscale=False,  # <-- Hide the colorbar
                    zmin=0, zmax=1,
                    text=heatmap_df.values.astype(str),
                    texttemplate="%{text}",
                )
            )
            # Detect dark mode and set background colors accordingly
            bg_color = st.get_option("theme.backgroundColor")
            
            fig.update_layout(
                xaxis_title="<b>Judge</b>",
                yaxis_title="<b>Subgoal</b>",
                font=dict(family="Georgia", size=14),
                title=dict(
                    text=f"<b>{trace} | {sample_label}</b>",
                    font=dict(size=16, color="#7882f1"),
                    x=0.5
                ),
                width=400,
                height=70 * max(3, len(heatmap_df)) + 120,
                margin=dict(l=20, r=20, t=60, b=20),
                plot_bgcolor=bg_color,
                paper_bgcolor=bg_color,
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(heatmap_df.index),
                    ticktext=[str(sg) for sg in heatmap_df.index]
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show pred_i, judge input, subgoal, and explanation for each judge for the selected subgoal in the heatmap
            st.markdown("**Judge Details:**")
            if not df_pair.empty:
                # Determine which subgoal to show details for
                if show_all_subgoals:
                    subgoal_options = subgoal_labels
                    subgoal_key = f"subgoal_detail_{trace}_{sample_idx}"
                    selected_subgoal_detail = st.selectbox(
                        "Select subgoal for judge details",
                        subgoal_options,
                        key=subgoal_key
                    )
                else:
                    selected_subgoal_detail = subgoal_labels[0] if subgoal_labels else None

                row = df_pair[df_pair["subgoal_idx"] == selected_subgoal_detail].iloc[0] if selected_subgoal_detail in df_pair["subgoal_idx"].values else df_pair.iloc[0]
                for i in range(num_judges):
                    pred_col = f"pred_{i}"
                    explanation = row["explanation"] if "explanation" in row else "(N/A)"
                    subgoal_val_from_df = row["subgoal_detail"] if "subgoal_detail" in row else "(N/A)"
                    subgoal_val_extracted = extract_subgoal_from_explanation(explanation=explanation) if isinstance(explanation, str) and explanation else "(N/A)"
                    if subgoal_val_from_df == "(N/A)":
                        real_subgoal_val = subgoal_val_extracted
                    else:
                        real_subgoal_val = subgoal_val_from_df
                    
                    pred_val = row[pred_col] if pred_col in row else "(N/A)"
                    judge_input = row["judge_model_input"] if "judge_model_input" in row else "(N/A)"
                    with st.expander(f"Judge {i+1}"):
                        st.markdown(f"- **Subgoal:** {real_subgoal_val}")
                        st.markdown(f"- **Judge Explanation:** {pred_val}")
                        show_input = st.checkbox(f"Show Judge Input for Judge {i+1}", key=f"show_input_{idx}_{i}")
                        if show_input:
                            st.code(judge_input)

