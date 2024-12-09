import argparse
import json
import math
import os
import yaml
from collections import defaultdict
from pathlib import Path
import shutil
from datetime import datetime


def collect_and_summarize_results(output_dir):
    results_summaries = defaultdict(list)

    # Collect per-episode results
    for env_name in os.listdir(output_dir):
        env_dir = os.path.join(output_dir, env_name)
        if not os.path.isdir(env_dir):
            continue

        # Recursively traverse directories under env_dir
        for root, dirs, files in os.walk(env_dir):
            for filename in files:
                if (
                    filename.endswith(".json")
                    and not filename.endswith("_summary.json")
                    and filename != "summary.json"
                ):
                    json_filepath = os.path.join(root, filename)
                    with open(json_filepath, "r") as f:
                        episode_log = json.load(f)
                        results_summaries[env_name].append(episode_log)

    # Summarize results per environment and overall
    overall_total_input_tokens = 0
    overall_total_output_tokens = 0
    overall_env_summaries = {}
    env_avg_progressions = []
    agent_config = None
    client_config = None
    config_collected = False

    print(f"Found results for {len(results_summaries)} environments.")

    for env_name, episodes in results_summaries.items():
        env_episode_progress = []
        env_total_steps = 0
        env_total_input_tokens = 0
        env_total_output_tokens = 0
        env_total_episodes = len(episodes)
        env_tasks = defaultdict(list)

        for episode_log in episodes:
            if (
                not config_collected
                and "client" in episode_log
                and "agent" in episode_log
            ):
                agent_config = episode_log["agent"]
                client_config = episode_log["client"]
                config_collected = True

            task_name = episode_log.get("task")
            env_tasks[task_name].append(episode_log)
            episode_progress = episode_log.get("progression", 0.0)
            env_episode_progress.append(episode_progress)
            env_total_steps += episode_log.get("num_steps", 0)
            env_total_input_tokens += episode_log.get("input_tokens", 0)
            env_total_output_tokens += episode_log.get("output_tokens", 0)

        # Calculate mean and standard error for the environment
        env_avg_progress = (
            sum(env_episode_progress) / env_total_episodes
            if env_total_episodes
            else 0.0
        )
        env_avg_progressions.append(env_avg_progress)
        env_std_dev = (
            math.sqrt(
                sum((x - env_avg_progress) ** 2 for x in env_episode_progress)
                / env_total_episodes
            )
            if env_total_episodes > 1
            else 0.0
        )
        env_std_error = (
            env_std_dev / math.sqrt(env_total_episodes)
            if env_total_episodes > 1
            else 0.0
        )

        # Update overall totals
        overall_total_input_tokens += env_total_input_tokens
        overall_total_output_tokens += env_total_output_tokens

        env_task_summaries = {}
        for task_name, task_runs in env_tasks.items():
            task_episode_progress = [run.get("progression", 0.0) for run in task_runs]
            task_count = len(task_runs)
            avg_task_progress = (
                sum(task_episode_progress) / task_count if task_count else 0.0
            )
            task_std_dev = (
                math.sqrt(
                    sum((x - avg_task_progress) ** 2 for x in task_episode_progress)
                    / task_count
                )
                if task_count > 1
                else 0.0
            )
            task_std_error = (
                task_std_dev / math.sqrt(task_count) if task_count > 1 else 0.0
            )

            env_task_summaries[task_name] = {
                "progression_percentage": 100 * avg_task_progress,
                "standard_error": 100 * task_std_error,
                "episodes_played": task_count,
            }

        avg_steps = env_total_steps / env_total_episodes if env_total_episodes else 0.0

        env_summary = {
            "progression_percentage": 100 * env_avg_progress,
            "standard_error": 100 * env_std_error,
            "average_steps": avg_steps,
            "episodes_played": env_total_episodes,
            "tasks": env_task_summaries,
            "input_tokens": env_total_input_tokens,
            "output_tokens": env_total_output_tokens,
        }

        # Save environment summary
        env_summary_filename = os.path.join(
            output_dir, env_name, f"{env_name}_summary.json"
        )
        Path(env_summary_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(env_summary_filename, "w") as f:
            json.dump(env_summary, f, indent=4)

        # Collect environment summaries for overall summary
        overall_env_summaries[env_name] = {
            "progression_percentage": env_summary["progression_percentage"],
            "standard_error": env_summary["standard_error"],
            "episodes_played": env_summary["episodes_played"],
        }

    # Now compute overall average progression as the mean of environment average progressions
    total_envs = len(env_avg_progressions)
    if total_envs > 0:
        overall_avg_progression = sum(env_avg_progressions) / total_envs
        # Collect per-environment standard errors
        env_standard_errors = [
            env_data["standard_error"] for env_data in overall_env_summaries.values()
        ]
        # Correctly calculate the combined standard error
        sum_of_squares = sum(se**2 for se in env_standard_errors)
        overall_std_error = math.sqrt(sum_of_squares) / total_envs
    else:
        overall_avg_progression = 0.0
        overall_std_error = 0.0

    summary = {
        "average_progress": 100 * overall_avg_progression,
        "standard_error": overall_std_error,
        "environments": overall_env_summaries,
        "total_input_tokens": overall_total_input_tokens,
        "total_output_tokens": overall_total_output_tokens,
        "client": client_config,
        "agent": agent_config,
    }

    # Save overall summary
    summary_filename = os.path.join(output_dir, "summary.json")
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_filename}")

    return summary


def print_summary_table(summary):
    print("\nSummary of Results:")
    print(
        f"Overall Average Progression: {summary['average_progress']:.2f}% ± {summary['standard_error']:.2f}%"
    )
    print("Per-Environment Results:")
    for env_name, env_data in summary["environments"].items():
        print(
            f"  {env_name}: {env_data['progression_percentage']:.2f}% ± {env_data['standard_error']:.2f}%, Episodes: {env_data['episodes_played']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize results and update data.")
    parser.add_argument(
        "path", nargs="?", default=None, help="Path to the submission directory."
    )
    args = parser.parse_args()

    submissions_dir = "submissions"
    leaderboards = ["LLM", "VLM"]

    if args.path is not None:
        # Process the submission at args.path
        output_dir = args.path
        summary = collect_and_summarize_results(output_dir)
        print_summary_table(summary)

    # Initialize the data structure
    data = {"leaderboards": []}

    # Iterate over each leaderboard
    for lb_name in leaderboards:
        lb_path = os.path.join(submissions_dir, lb_name)
        lb_entry = {"name": lb_name, "results": []}

        # Check if the leaderboard directory exists
        if os.path.isdir(lb_path):
            # List all submissions in the leaderboard directory
            submissions = [
                sub
                for sub in os.listdir(lb_path)
                if os.path.isdir(os.path.join(lb_path, sub))
            ]

            # Iterate over each submission
            for submission in submissions:
                submission_path = os.path.join(lb_path, submission)
                summary_path = os.path.join(submission_path, "summary.json")

                # Check if summary.json exists in the submission directory
                if os.path.isfile(summary_path):
                    # Read the summary.json file
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                    # Read the metadata.yaml file
                    metadata_path = os.path.join(submission_path, "metadata.yaml")
                    if os.path.isfile(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = yaml.safe_load(f)
                    else:
                        metadata = (
                            {}
                        )  # Use an empty dict if metadata.yaml does not exist

                    # Prepare the result entry with default values
                    envs = summary.get("environments", {})

                    results_summary = {}
                    for env_name, env in envs.items():
                        results_summary[env_name] = [
                            env["progression_percentage"],
                            env["standard_error"],
                            env["episodes_played"],
                        ]

                    results_summary["average"] = [
                        summary.get("average_progress", 0.0),
                        summary.get("standard_error", 0.0),
                    ]

                    # Extract date from submission_path
                    # Try to get date from metadata, or use modification time
                    date = metadata.get("date")
                    if not date:
                        try:
                            mtime = os.path.getmtime(summary_path)
                            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                        except Exception as e:
                            date = ""

                    metadata_entry = {
                        "name": metadata.get("name", ""),
                        "folder": submission_path,
                        "date": date,
                        "trajs": metadata.get("trajs", ""),
                        "site": metadata.get("site", ""),
                        "verified": metadata.get("verified", False),
                        "oss": metadata.get("oss", False),
                        "org_logo": metadata.get("org_logo", ""),
                    }

                    result_entry = {**results_summary, **metadata_entry}

                    # Append the result entry to the leaderboard results
                    lb_entry["results"].append(result_entry)
                else:
                    print(f"Warning: 'summary.json' not found in {submission_path}")
        else:
            print(f"Warning: Leaderboard directory '{lb_name}' does not exist.")

        # Append the leaderboard entry to the data
        data["leaderboards"].append(lb_entry)

    # Write the compiled data to data.json
    os.makedirs("template", exist_ok=True)
    with open("template/data.json", "w") as outfile:
        json.dump(data, outfile, indent=2)
    print("Leaderboard data updated in 'template/data.json'")


if __name__ == "__main__":
    main()
