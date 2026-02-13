import pandas as pd

from typing import List

from agent_inspect.models.metrics import SubGoal, SubGoalValidationResult
from agent_inspect.models.tools import ErrorAnalysisDataSample


def get_score_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith('pred_') and col.endswith('_score')]


def get_explanation_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith('pred_') and not col.endswith('_score')]


def convert_df_to_error_analysis_data_samples(df: pd.DataFrame) -> List[ErrorAnalysisDataSample]:
    """
    Convert a DataFrame of subgoal validation rows into ErrorAnalysisDataSample objects.
    Rows are grouped by sample_idx and converted into structured data samples.
    """
    if df.empty:
        return []

    # Extract data_sample_id from sample_idx
    working_df = df.copy()
    working_df["data_sample_id"] = (
        working_df["sample_idx"]
        .astype(str)
        .str.removeprefix("multiturnmessage_")
        .astype(int)
    )

    score_cols = get_score_columns(working_df)
    explanation_cols = get_explanation_columns(working_df)
    data_samples: List[ErrorAnalysisDataSample] = []
    for datasample_id, datasample_group in working_df.groupby("data_sample_id", sort=False):
        for agent_run_id, agent_run_group in datasample_group.groupby("agent_run_id", sort=False):
            subgoal_validations: List[SubGoalValidationResult] = []
            for _, row in agent_run_group.iterrows():
                # Extract judge trials and prepend placeholder summary (expected by error analysis)
                judge_trials_explanations = row[explanation_cols].to_list()
                SUMMARY_PLACEHOLDER = "DUMMY STRING"
                explanations_with_summary = [SUMMARY_PLACEHOLDER] + judge_trials_explanations

                subgoal_validations.append(
                    SubGoalValidationResult(
                        is_completed=subgoal_is_completed(row[score_cols].to_list()),
                        explanations=explanations_with_summary,
                        sub_goal=SubGoal(details=row["subgoal_detail"]),
                        prompt_sent_to_llmj=row["judge_model_input"]
                    )
                )

            data_samples.append(
                ErrorAnalysisDataSample(
                    data_sample_id=datasample_id,
                    subgoal_validations=subgoal_validations,
                    agent_run_id=agent_run_id
                )
            )

    return data_samples


def subgoal_is_completed(pred_scores: List[int]) -> bool:
    """
    Determine if a subgoal validation is completed based on prediction scores.
    A subgoal is considered completed if more than half of prediction scores are 1.
    """
    if not pred_scores:
        return False
    positive_votes = sum(1 for score in pred_scores if score == 1)
    return positive_votes >= len(pred_scores) // 2 + 1


def select_n_agent_runs(df: pd.DataFrame, number_of_agent_runs: int) -> pd.DataFrame:
    """
    Select a subset of agent runs from the DataFrame based on unique sample_idx and subgoal_idx.
    Add a new column agent_run_id to identify different agent runs.
    """
    # Find unqiue filenames and select the first N unique ones
    unique_filenames = df["filename"].unique()[:number_of_agent_runs]
    filtered_df = df[df["filename"].isin(unique_filenames)].reset_index(drop=True)

    # Map filenames to agent_run_ids
    filename_to_run_id = {filename: run_id for run_id, filename in enumerate(unique_filenames)}
    filtered_df["agent_run_id"] = filtered_df["filename"].map(filename_to_run_id)

    return filtered_df


def load_error_analysis_data(data_csv_path, number_of_agent_runs=4) -> List[ErrorAnalysisDataSample]:
    """Load error analysis data from CSV."""
    df = pd.read_csv(data_csv_path)

    # Choose a subset of agent runs for testing
    df = select_n_agent_runs(df, number_of_agent_runs)

    print(f"Loaded {len(df)} rows from csv for testing")

    # Drop unneeded columns
    relevant_cols = ["sample_idx", "subgoal_detail", "agent_run_id", "judge_model_input"] + get_score_columns(
        df) + get_explanation_columns(df)
    df = df[relevant_cols]

    list_of_data_samples = convert_df_to_error_analysis_data_samples(df)

    return list_of_data_samples
