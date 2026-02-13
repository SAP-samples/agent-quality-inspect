# Load and view results in Streamlit
import argparse
import pickle
from demo.ui_for_agent_diagnosis.app import launch_ui


def parse_args():
    parser = argparse.ArgumentParser(description="View agent evaluation error analysis results in Streamlit UI.")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output folder to view error analysis results from.",
    )

    return parser.parse_args()

args = parse_args()

with open(f"{args.output_dir}/error_analysis.pkl", "rb") as f:
    error_analysis_data_samples, error_analysis_results = pickle.load(f)

launch_ui(error_analysis_results, error_analysis_data_samples)
