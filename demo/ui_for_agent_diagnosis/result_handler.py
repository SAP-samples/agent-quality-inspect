"""
Handler class to process ErrorAnalysisResult and StatisticAnalysisResult
for Streamlit UI visualization.
"""
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from agent_inspect.models.tools import (
    ErrorAnalysisResult, 
    StatisticAnalysisResult,
    AnalyzedSubgoalValidation
)
from agent_inspect.metrics.utils.metrics_utils import tally_votes


class ResultHandler:
    """
    Processes ErrorAnalysisResult and StatisticAnalysisResult objects
    and prepares data for Streamlit UI visualization.
    """
    
    def __init__(
        self, 
        error_analysis_result: ErrorAnalysisResult,
        statistic_analysis_results: List[StatisticAnalysisResult]
    ):
        """
        Initialize the handler with analysis results.
        
        Args:
            error_analysis_result: The error analysis result containing clustered errors
            statistic_analysis_results: List of statistic analysis results for each data sample
        """
        self.error_analysis_result = error_analysis_result
        self.statistic_analysis_results = statistic_analysis_results
        
        # Create a mapping from (agent_run_id, data_sample_id) to StatisticAnalysisResult
        self.stats_map = {}
        for stat_result in statistic_analysis_results:
            key = (stat_result.agent_run_id, stat_result.data_sample_id)
            self.stats_map[key] = stat_result
        
        logging.info(f"ResultHandler initialized with {len(statistic_analysis_results)} statistic results")
    
    def prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare two DataFrames needed by the UI:
        1. mean_df: Contains mean expectation and std for each (agent_run_id, sample_idx) pair
        2. full_df: Contains detailed subgoal-level information
        
        Returns:
            Tuple of (mean_df, full_df)
        """
        mean_rows: List[Dict[str, Any]] = []
        full_rows: List[Dict[str, Any]] = []
        
        # Track subgoal indices per (agent_run_id, sample_id)
        sample_subgoal_dict: Dict[Tuple, Dict[str, int]] = defaultdict(dict)
        
        # Process clustered errors
        for cluster_label, analyzed_validations in self.error_analysis_result.analyzed_validations_clustered_by_errors.items():
            for analyzed_validation in analyzed_validations:
                self._process_analyzed_validation(
                    analyzed_validation,
                    cluster_label,
                    analyzed_validation.base_error,
                    full_rows,
                    sample_subgoal_dict
                )
        
        # Process completed subgoals (no errors)
        for completed_validation in self.error_analysis_result.completed_subgoal_validations:
            self._process_analyzed_validation(
                completed_validation,
                None,  # No cluster label
                None,  # No base error
                full_rows,
                sample_subgoal_dict
            )
        
        # Create full_df
        full_df = pd.DataFrame(full_rows)
        if not full_df.empty:
            # print("Full DF Columns:", full_df.columns)
            full_df.sort_values(by=["agent_run_id", "sample_idx", "subgoal_idx"], inplace=True)

            # Re-sort subgoal_idx for each sample_idx by subgoal_detail alphabetically, consistent across agent_run_id
            def reassign_subgoal_idx(df):
                # For each sample_idx, get the sorted subgoal_detail order
                subgoal_order = (
                    df.groupby("sample_idx")
                    ["subgoal_detail"]
                    .unique()
                    .apply(lambda x: sorted(list(x)))
                )
                # Build a mapping: original sample_idx -> subgoal_detail -> new_idx
                subgoal_idx_map = {
                    sample_idx: {detail: idx+1 for idx, detail in enumerate(details)}
                    for sample_idx, details in subgoal_order.items()
                }
                # Apply the mapping
                def get_new_idx(row):
                    return subgoal_idx_map[row["sample_idx"]][row["subgoal_detail"]]
                df["subgoal_idx"] = df.apply(get_new_idx, axis=1)
                return df
            full_df = reassign_subgoal_idx(full_df)
        
        # Create mean_df from statistic analysis results
        for stat_result in self.statistic_analysis_results:
            agent_run_id = stat_result.agent_run_id if stat_result.agent_run_id is not None else "default_run"
            sample_idx = stat_result.data_sample_id
            
            mean_rows.append({
                "agent_run_id": str(agent_run_id),
                "sample_idx": sample_idx,
                "SB_token_prob_mean": stat_result.judge_expectation,
                "SB_token_prob_SD_aggregated": stat_result.judge_std
            })
        
        mean_df = pd.DataFrame(mean_rows)
        if not mean_df.empty:
            mean_df.sort_values(by=["agent_run_id", "sample_idx"], inplace=True)
        
        logging.info(f"Prepared mean_df with {len(mean_df)} rows and full_df with {len(full_df)} rows")
        
        return mean_df, full_df
    
    def _process_analyzed_validation(
        self,
        analyzed_validation: AnalyzedSubgoalValidation,
        cluster_label: Optional[str],
        base_error: Optional[str],
        full_rows: List[Dict[str, Any]],
        sample_subgoal_dict: Dict[Tuple, Dict[str, int]]
    ):
        """
        Process a single AnalyzedSubgoalValidation and add rows to full_rows.
        """
        agent_run_id = analyzed_validation.agent_run_id if analyzed_validation.agent_run_id is not None else "default_run"
        sample_id = analyzed_validation.data_sample_id
        subgoal_detail = analyzed_validation.subgoal_validation.sub_goal.details
        
        # Track subgoal index
        key = (agent_run_id, sample_id)
        if subgoal_detail not in sample_subgoal_dict[key]:
            subgoal_idx = len(sample_subgoal_dict[key]) + 1
            sample_subgoal_dict[key][subgoal_detail] = subgoal_idx
        else:
            subgoal_idx = sample_subgoal_dict[key][subgoal_detail]
        
        # Build row data
        row = {
            "agent_run_id": str(agent_run_id),
            "sample_idx": sample_id,
            "subgoal_idx": int(subgoal_idx),
            "subgoal_detail": subgoal_detail,
            "judge_model_input": analyzed_validation.subgoal_validation.prompt_sent_to_llmj
        }
        
        # Add cluster label and base error if present
        if cluster_label is not None:
            row["cluster_label"] = cluster_label
        if base_error is not None:
            row["final_error_type"] = base_error
        
        # Extract judge scores from explanations (skip first summarized explanation)
        explanations = analyzed_validation.subgoal_validation.explanations[1:]
        for i, explanation in enumerate(explanations):
            if explanation == "DUMMY STRING":
                continue
            
            row[f"pred_{i}"] = explanation
            # Calculate score: 1.0 if 'C' (completed), 0.0 otherwise
            c_count = tally_votes(0, 0, 0, [explanation])[0]
            row[f"pred_{i}_score"] = 1.0 if c_count == 1 else 0.0
        
        full_rows.append(row)
    
    def get_ui_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Public method to get the prepared DataFrames for UI.
        
        Returns:
            Tuple of (mean_df, full_df)
        """
        return self.prepare_dataframes()
