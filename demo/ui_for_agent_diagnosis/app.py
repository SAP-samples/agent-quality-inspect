"""
Automated Error Analysis Tool - Streamlit UI Application

This module provides visualizing error analysis results with in-memory data or CSV files (legacy method).
"""
import os

import streamlit as st

from typing import Optional, List
from demo.ui_for_agent_diagnosis.features.consistency_viz import consistency_viz_ui
from demo.ui_for_agent_diagnosis.result_handler import ResultHandler
from agent_inspect.models.tools import (
    ErrorAnalysisResult,
    StatisticAnalysisResult,
    ErrorAnalysisDataSample
)
from agent_inspect.tools import StatisticAnalysis


def main(
    error_analysis_result: Optional[ErrorAnalysisResult] = None,
    statistic_analysis_results: Optional[List[StatisticAnalysisResult]] = None
):
    """
    Main application entry point.
    
    Args:
        error_analysis_result: In-memory ErrorAnalysisResult object (optional)
        statistic_analysis_results: In-memory list of StatisticAnalysisResult objects (optional)
    
    If in-memory data is provided, it will be used directly.
    Otherwise, the app will fall back to static CSV file upload. We provide a default sample CSV for convenience.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_df_path = os.path.join(base_dir, "examples/sample_final_error_summary.csv")

    st.sidebar.header("Select a Feature")
    feature = st.sidebar.selectbox("Choose a feature", ["Automated Error Analysis"])

    st.title("Automated Error Analysis Tool")
    
    if feature == "Automated Error Analysis":
        # Check if in-memory data is provided
        if error_analysis_result is not None and statistic_analysis_results is not None:
            st.sidebar.success("Using in-memory data")
            # Process in-memory data
            handler = ResultHandler(error_analysis_result, statistic_analysis_results)
            mean_df, full_df = handler.get_ui_data()
            consistency_viz_ui(mean_df=mean_df, full_df=full_df)
        else:
            # Fall back to CSV file upload
            st.sidebar.header("Data Selection")
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file to analyse",
                type=["csv"],
                help="Upload your CSV file for analysis. If not uploaded, the default sample data will be used."
            )

            if uploaded_file is not None:
                df_path = uploaded_file
            else:
                df_path = default_df_path
            
            consistency_viz_ui(csv_path=df_path)


def launch_ui(
    error_analysis_result: ErrorAnalysisResult,
    data_samples: List[ErrorAnalysisDataSample]
) -> None:
    """
    Launch the Streamlit UI with error analysis results. This function computes statistical analysis internally using StatisticAnalysis. 
    
    It will compute the statistical analysis for each data sample first, then launches the Streamlit UI to visualize the results.
    
    Args:
        error_analysis_result: The error analysis result containing clustered errors
        data_samples: List of data samples used for error analysis
    
    Usage:
        `your_script.py` snippet:
        
        ```python
        from agent_inspect.tools.ui_for_agent_diagnosis.app import launch_ui

        # Prepare your data
        error_result = ErrorAnalysisResult(...)
        data_samples = [ErrorAnalysisDataSample(...), ...]

        # Launch the UI
        launch_ui(error_result, data_samples)
        ```

        Then run: `streamlit run your_script.py`
    
    """
    # Compute statistical analysis for each data sample
    statistic_results = []
    
    for data_sample in data_samples:
        stat_result = StatisticAnalysis.compute_statistic_analysis_result(data_sample)
        statistic_results.append(stat_result)
    
    # Launch the UI with in-memory data
    main(
        error_analysis_result=error_analysis_result,
        statistic_analysis_results=statistic_results
    )


def launch_ui_with_stats(
    error_analysis_result: ErrorAnalysisResult,
    statistic_analysis_results: List[StatisticAnalysisResult]
) -> None:
    """
    Launch the Streamlit UI with pre-computed statistical analysis results.
    
    Use this function if you have already computed the statistical analysis somewhere else
    and want to directly visualize the results.
    
    Args:
        error_analysis_result: The error analysis result containing clustered errors
        statistic_analysis_results: Pre-computed list of statistical analysis results
    
    Usage:
        `your_script.py` snippet:
        
        ```python
        from agent_inspect.tools.ui_for_agent_diagnosis.app import launch_ui_with_stats
        
        # Prepare your data
        error_result = ErrorAnalysisResult(...)
        stat_results = [StatisticAnalysisResult(...), ...]
        
        # Launch the UI
        launch_ui_with_stats(error_result, stat_results)
        ```
        
        Then run: streamlit run your_script.py
    """
    main(
        error_analysis_result=error_analysis_result,
        statistic_analysis_results=statistic_analysis_results
    )


if __name__ == "__main__":
    # Example steps: 
    #   1. Load from error_analysis_result.pkl for testing, which contains `ErrorAnalysisResult` object
    #   2. Load data samples from final_error_summary.csv for testing, which will convert to `List[ErrorAnalysisDataSample]` objects
    
    import pickle
    from utils import load_error_analysis_data
    
    PATH_TO_ERROR_SUMMARY_CSV = "tests_acceptance/tools/sample_data/final_error_summary.csv"
    NUMBER_OF_AGENT_RUNS = 6
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(base_dir, "examples/error_analysis_result.pkl")
    
    # Try to load example data if available
    if os.path.exists(example_path):
        try:
            with open(example_path, "rb") as f:
                error_result: ErrorAnalysisResult = pickle.load(f)
            
            data_samples: List[ErrorAnalysisDataSample] = load_error_analysis_data(PATH_TO_ERROR_SUMMARY_CSV, NUMBER_OF_AGENT_RUNS)
            
            # Launch with example data
            launch_ui(error_result, data_samples)
        except Exception as e:
            print(f"Could not load example data: {e}")
            print("Falling back to CSV upload mode...")
            main()
    else:
        # Fall back to legacy CSV file upload mode
        main()