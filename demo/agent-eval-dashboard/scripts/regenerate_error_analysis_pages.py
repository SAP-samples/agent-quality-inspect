"""
Regenerate all error analysis pages for leaderboard entries.

This script finds all experiment output folders with error_analysis.pkl files
and regenerates the error analysis HTML pages using the improved v2 generator.

Usage:
    python demo/agent-eval-dashboard/scripts/regenerate_error_analysis_pages.py
"""

import json
import sys
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from page_generators.generate_error_analysis_pages import generate_error_analysis_pages

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DASHBOARD_ROOT = Path(__file__).parent.parent  # demo/agent-eval-dashboard
LEADERBOARD_DATA = DASHBOARD_ROOT / "leaderboard_data.json"


def main():
    """Regenerate all error analysis pages."""

    # Load leaderboard data
    if not LEADERBOARD_DATA.exists():
        print(f"Error: {LEADERBOARD_DATA} not found")
        return

    with open(LEADERBOARD_DATA, "r") as f:
        data = json.load(f)

    datasets = data.get("datasets", {})

    if not datasets:
        print("No datasets found in leaderboard_data.json")
        return

    # Collect all entries from all datasets
    all_entries = []
    for dataset_key, dataset_data in datasets.items():
        dataset_name = dataset_data.get("display_name", dataset_key)
        entries = dataset_data.get("entries", [])

        for entry in entries:
            # Add dataset info to entry
            entry["dataset_name"] = dataset_name
            entry["dataset_key"] = dataset_key
            all_entries.append(entry)

    if not all_entries:
        print("No entries found in any dataset")
        return

    print(f"Found {len(all_entries)} entries across {len(datasets)} datasets")
    print("=" * 80)

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, entry in enumerate(all_entries, 1):
        model = entry.get("agent_model", "unknown")
        dataset = entry.get("dataset_name", "unknown")
        source_folder = Path(entry.get("source_folder", ""))

        print(f"\n[{i}/{len(all_entries)}] Processing: {model}")
        print(f"  Dataset: {dataset}")
        print(f"  Source: {source_folder.name if source_folder else 'N/A'}")

        if not source_folder or not source_folder.exists():
            print("  [skip] Source folder not found")
            skip_count += 1
            continue

        # Check if error_analysis.pkl exists
        pkl_path = source_folder / "error_analysis.pkl"
        if not pkl_path.exists():
            print("  [skip] No error_analysis.pkl")
            skip_count += 1
            continue

        try:
            result = generate_error_analysis_pages(entry, dataset, source_folder)
            if result:
                print(f"  ✓ Generated: {result.relative_to(REPO_ROOT)}")
                success_count += 1
            else:
                print("  [skip] Generation returned None")
                skip_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Successfully generated: {success_count}")
    print(f"  Skipped (no data): {skip_count}")
    print(f"  Errors: {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
