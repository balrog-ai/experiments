import os
import json
import yaml

# Define the base directory for submissions
submissions_dir = "submissions"

# Define the leaderboards
leaderboards = ["LLM", "VLM"]

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
                    summary.get("Final score", []),
                    summary.get("standard_error", []),
                ]

                date = submission_path.split("_")[0].split("/")[-1]
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

                metadata = {
                    "name": metadata.get("name", ""),
                    "folder": submission_path,
                    "date": date,
                    "trajs": metadata.get("trajs", ""),
                    "site": metadata.get("site", ""),
                    "verified": metadata.get("verified", False),
                    "oss": metadata.get("oss", False),
                    "org_logo": metadata.get("org_logo", ""),
                }

                result_entry = {**results_summary, **metadata}

                # Append the result entry to the leaderboard results
                lb_entry["results"].append(result_entry)
            else:
                print(f"Warning: 'summary.json' not found in {submission_path}")
    else:
        print(f"Warning: Leaderboard directory '{lb_name}' does not exist.")

    # Append the leaderboard entry to the data
    data["leaderboards"].append(lb_entry)

# Write the compiled data to data.json
with open("data.json", "w") as outfile:
    json.dump(data, outfile, indent=2)
