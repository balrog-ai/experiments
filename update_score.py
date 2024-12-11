import json
import math
import argparse


def recalculate_summary_stats(summary_data):
    envs = summary_data.get("environments", {})
    if not envs:
        # No environments; can't recalculate meaningful statistics
        summary_data["average_progress"] = 0.0
        summary_data["standard_error"] = 0.0
        return summary_data

    # Extract progression percentages and standard errors
    progression_percentages = [env["progression_percentage"] for env in envs.values()]
    std_errors = [env["standard_error"] for env in envs.values()]

    # Calculate the new average progression
    total_envs = len(progression_percentages)
    average_progress = sum(progression_percentages) / total_envs

    # Calculate combined standard error using RMS of individual standard errors
    sum_of_squares = sum((se**2) for se in std_errors)
    combined_std_error = math.sqrt(sum_of_squares) / total_envs

    summary_data["average_progress"] = average_progress
    summary_data["standard_error"] = combined_std_error

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recalculate and update summary statistics in summary.json"
    )
    parser.add_argument("summary_path", help="Path to the summary.json file")
    args = parser.parse_args()

    summary_file = args.summary_path

    # Load existing summary
    with open(summary_file, "r") as f:
        summary = json.load(f)

    # Recalculate and update
    summary = recalculate_summary_stats(summary)

    # Save updated summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Updated 'average_progress' to {summary['average_progress']}")
    print(f"Updated 'standard_error' to {summary['standard_error']}")
