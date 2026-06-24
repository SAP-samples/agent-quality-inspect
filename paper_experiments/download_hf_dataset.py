"""Download from HuggingFace — supports both dataset and model repos."""

import argparse
from huggingface_hub import hf_hub_download, snapshot_download


def download_files():
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset or model repo.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="SAP/agent-quality-inspect",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./downloads",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Download a specific file instead of the full repo",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["model", "dataset"],
        help="Type of repository (model or dataset)",
    )
    args = parser.parse_args()

    try:
        if args.filename:
            print(f"Downloading file '{args.filename}' from {args.repo_id} to {args.output_dir}")
            path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                local_dir=args.output_dir,
                repo_type=args.repo_type,
            )
            print(f"Downloaded to: {path}")
        else:
            print(f"Downloading {args.repo_type} repo: {args.repo_id} to {args.output_dir}")
            path = snapshot_download(
                repo_id=args.repo_id,
                local_dir=args.output_dir,
                repo_type=args.repo_type,
            )
            print(f"Download successful! Saved to: {path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print()
        print("If authentication is required, log in first:")


if __name__ == "__main__":
    download_files()
