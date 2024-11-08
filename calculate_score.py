import json
import math


def update_final_score(filename):
    # Read the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Get the environments data
    environments = data.get("environments", {})

    # Initialize lists to store progression percentages and standard errors
    progression_percentages = []
    standard_errors = []

    for env_data in environments.values():
        pp = env_data.get("progression_percentage", 0.0)
        se = env_data.get("standard_error", 0.0)
        progression_percentages.append(pp)
        standard_errors.append(se)

    # Calculate the averages
    num_envs = len(progression_percentages)
    if num_envs > 0:
        final_score = sum(progression_percentages) / num_envs
        # Combined standard error
        sum_of_squares = sum(se**2 for se in standard_errors)
        final_standard_error = (1 / num_envs) * math.sqrt(sum_of_squares)
    else:
        final_score = 0.0
        final_standard_error = 0.0

    # Update the JSON data
    data["average_progress"] = final_score
    data["standard_error"] = final_standard_error

    # Write back to the JSON file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The name of the JSON file to update")
    args = parser.parse_args()

    update_final_score(args.filename)
