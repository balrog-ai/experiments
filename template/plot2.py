import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# Define the models to include and their corresponding colors
model_colors = {
    "claude-3.5-sonnet": "#c20202",
    "gpt-4o-mini": "#ff8700",
    "gpt-4o": "#fbcd2b",
    "gemini-1.5-flash": "#00ff5c",
    "gemini-1.5-pro": "#00ffdb",
    "llama-3.2-1b-it": "#28befa",
    "llama-3.2-3b-it": "#007eff",
    "llama-3.1-8b-it": "#0200ff",
    "llama-3.1-70b-it": "#8200ff",
    "llama-3.2-11b-it": "#ff00fd",
    "llama-3.2-90b-it": "#ff007e",
    # Add more models and their colors if needed
}

# Load your data here
# Replace this placeholder with your actual data loading code
# For example, you might load data from a JSON file
# with open('your_data_file.json', 'r') as f:
#     data = json.load(f)

# Example placeholder for data structure
# Please replace this with your actual data
data = {
    "leaderboards": [
        {
            "name": "LLM",
            "results": [
                {
                    "name": "gpt-4o",
                    "TextWorld": (85.0, 2.0, 10),
                    "GameA": (90.0, 1.5, 10),
                    "GameB": (80.0, 1.0, 10),
                    # Add other games and their results
                },
                # Add other LLM models and their results
            ],
        },
        {
            "name": "VLM",
            "results": [
                {
                    "name": "gpt-4o",
                    # VLMs don't have TextWorld data; it will be set to zero later
                    "GameA": (75.0, 2.5, 10),
                    "GameB": (70.0, 1.8, 10),
                    # Add other games and their results
                },
                # Add other VLM models and their results
            ],
        },
        # Include other modalities if necessary
    ]
}

# Extract the list of models to include from the keys of model_colors
include_models = list(model_colors.keys())

# Process data into 'results' dictionary
results = {"llm": {}, "vlm": {}}

game_names_set = set()

for leaderboard in data["leaderboards"]:
    modality = leaderboard["name"].lower()  # 'llm' or 'vlm'
    for model_result in leaderboard["results"]:
        model_name = model_result["name"].lower()
        if model_name not in include_models:
            continue  # Skip models not in the include_models list
        for game in model_result:
            if game in [
                "name",
                "average",
                "folder",
                "date",
                "trajs",
                "site",
                "verified",
                "oss",
                "org_logo",
            ]:
                continue
            value, error, count = model_result[game]
            game_names_set.add(game)
            if game not in results[modality]:
                results[modality][game] = {}
            results[modality][game][model_name] = (value, error)

# Ensure that all tasks are present in both modalities and for all models
all_tasks = sorted(list(game_names_set))
for modality in ["llm", "vlm"]:
    for task in all_tasks:
        if task not in results[modality]:
            results[modality][task] = {}
        for model in include_models:
            if model not in results[modality][task]:
                results[modality][task][model] = (0, 0)  # Set missing data to zero


# Function to plot average progression for LLM and VLM with hatch over colored bars
def plot_average_progression(results, title, filename, model_colors):
    include_models = list(model_colors.keys())

    # Collect all tasks
    all_tasks = set()
    for modality in results:
        for task in results[modality]:
            all_tasks.add(task)
    all_tasks = sorted(list(all_tasks))

    # Initialize data storage for averages
    avg_progression = {"llm": {}, "vlm": {}}

    # Compute averages for LLM and VLM over the same tasks
    for modality in ["llm", "vlm"]:
        for model in include_models:
            values = []
            for task in all_tasks:
                if model in results[modality][task]:
                    value = results[modality][task][model][
                        0
                    ]  # Extract progression value
                    values.append(value)
            if values:
                avg_progression[modality][model] = np.mean(values)
            else:
                avg_progression[modality][model] = 0  # Set to 0 if no data

    # Prepare data for plotting
    models = include_models
    x = np.arange(len(models))
    bar_width = 0.35

    llm_values = [avg_progression["llm"].get(model, 0) for model in models]
    vlm_values = [avg_progression["vlm"].get(model, 0) for model in models]
    colors = [model_colors.get(model, "#000000") for model in models]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot LLM bars
    ax.bar(
        x - bar_width / 2,
        llm_values,
        width=bar_width,
        color=colors,
        edgecolor="black",
        label="_nolegend_",  # Exclude these bars from the legend
        capsize=5,
    )

    # Plot VLM bars with hatch pattern over the colored bars
    ax.bar(
        x + bar_width / 2,
        vlm_values,
        width=bar_width,
        color=colors,
        edgecolor="black",  # Hatch color
        hatch="//",
        label="_nolegend_",  # Exclude these bars from the legend
        capsize=5,
    )

    # Create custom legend handles with grey color
    grey_patch = mpatches.Patch(facecolor="grey", edgecolor="black", label="LLM")
    grey_hatched_patch = mpatches.Patch(
        facecolor="grey", edgecolor="black", hatch="//", label="VLM"
    )

    # Adjust x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=12)
    ax.set_title(title, fontsize=22)
    ax.set_ylabel("Average Progress (%)", fontsize=16)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7)
    ax.set_yticks(np.arange(0, 101, 10))  # Add lines for every 10% increment

    # Use the custom legend handles
    ax.legend(handles=[grey_patch, grey_hatched_patch], fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# Plot and save the average progression results
plot_average_progression(
    results,
    "Average Progression for LLM and VLM",
    "average_progression.png",
    model_colors,
)
