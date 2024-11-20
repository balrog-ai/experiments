import matplotlib.pyplot as plt
import numpy as np
import json

# Set Helvetica as the default font for all text in the plots
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = "Helvetica"  # Use Helvetica as sans-serif option

# Increase font sizes for various plot elements
plt.rcParams["font.size"] = 14  # Default font size for all elements
plt.rcParams["axes.titlesize"] = 20  # Font size for subplot titles
plt.rcParams["axes.labelsize"] = 18  # Font size for x and y labels
plt.rcParams["xtick.labelsize"] = 18  # Font size for x tick labels
plt.rcParams["ytick.labelsize"] = 20  # Font size for y tick labels
plt.rcParams["figure.titlesize"] = 20  # Font size for the main figure title
plt.rcParams["legend.fontsize"] = 16  # Font size for legend text (if used)

# Read data from data.json
with open("data.json", "r") as f:
    data = json.load(f)

# Define the models to include and their corresponding colors
model_colors = {
    "gpt-4o-mini": "#ff8700",
    "gpt-4o": "#fbcd2b",
    "gemini-1.5-flash": "#00ff5c",
    "gemini-1.5-pro": "#00ffdb",
    "llama-3.2-1b-it": "#00fdff",
    "llama-3.2-3b-it": "#007eff",
    "llama-3.1-8b-it": "#0200ff",
    "llama-3.1-70b-it": "#8200ff",
    "llama-3.2-11b-it": "#ff00fd",
    "llama-3.2-90b-it": "#ff007e",
    "claude-3.5-sonnet": "#f08258",
    # "o1-preview": ,
}

# model_colors = {
#     "gpt-4o-mini": "#45e150",
#     "gpt-4o": "#45e19e",
#     "gemini-1.5-flash": "#45d6e1",
#     "gemini-1.5-pro": "#4588e1",
#     "llama-3.2-1b-it": "#5045e1",
#     "llama-3.2-3b-it": "#9e45e1",
#     "llama-3.1-8b-it": "#eb3840",
#     "llama-3.1-70b-it": "#eb8938",
#     "llama-3.2-11b-it": "#ebe338",
#     "llama-3.2-90b-it": "#9aeb38",
#     "claude-3.5-sonnet": "#41eb38",
#     # "o1-preview": ,
# }


# Extract the list of models to include from the keys of model_colors
include_models = list(model_colors.keys())
# print(include_models)

# Process data into 'results' dictionary
results = {"llm": {}, "vlm": {}}

game_names_set = set()

for leaderboard in data["leaderboards"]:
    modality = leaderboard["name"].lower()  # 'llm' or 'vlm'
    for model_result in leaderboard["results"]:
        model_name = model_result["name"].lower()
        if model_name not in include_models:
            print(include_models)
            print(model_name)
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


# print(results)

# Define game ordering and display names
game_ordering = {
    "babyai": "BabyAI",
    "crafter": "Crafter",
    "textworld": "TextWorld",
    "babaisai": "BabaIsAI",
    "minihack": "MiniHack",
    "nle": "NetHack",
}


# Function to plot combined LLM and VLM results with consistent model ordering
def plot_combined_results(results, title, filename, game_ordering, model_colors):
    include_models = list(model_colors.keys())
    # Filter tasks based on the ordering and data availability
    tasks = [
        task
        for task in game_ordering.keys()
        if task in results["llm"] or task in results["vlm"]
    ]

    # Adjusted subplot arrangement
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15), sharey=True)
    fig.suptitle(title, fontsize=22)
    axes = axes.flatten()

    plt.subplots_adjust(wspace=-1.0, hspace=0.4)

    for ax, task in zip(axes, tasks):
        models_llm = results["llm"].get(task, {})
        models_vlm = results["vlm"].get(task, {})

        # Get the list of models present in this task and in include_models
        models_in_task = (set(models_llm.keys()) | set(models_vlm.keys())) & set(
            include_models
        )
        # Order model_names according to the include_models list
        desired_model_order = [
            model for model in include_models if model in models_in_task
        ]

        num_models = len(desired_model_order)
        x = np.arange(num_models)
        bar_width = 0.35  # Adjust as needed

        # Extract llm and vlm values and errors
        llm_values = []
        llm_errors = []
        vlm_values = []
        vlm_errors = []
        colors = []

        for model in desired_model_order:
            colors.append(model_colors.get(model, "#000000"))
            # Get llm data
            if model in models_llm:
                llm_values.append(models_llm[model][0])
                llm_errors.append(models_llm[model][1])
            else:
                llm_values.append(0)
                llm_errors.append(0)
            # Get vlm data
            if model in models_vlm:
                vlm_values.append(models_vlm[model][0])
                vlm_errors.append(models_vlm[model][1])
            else:
                vlm_values.append(0)
                vlm_errors.append(0)

        # Plot llm bars
        ax.bar(
            x - bar_width / 2,
            llm_values,
            width=bar_width,
            yerr=llm_errors,
            capsize=5,
            color=colors,
            label="LLM" if ax == axes[0] else "",
        )
        # Plot vlm bars with hatching
        ax.bar(
            x + bar_width / 2,
            vlm_values,
            width=bar_width,
            yerr=vlm_errors,
            capsize=5,
            color=colors,
            hatch="//",
            label="VLM" if ax == axes[0] else "",
        )

        # Adjust x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(desired_model_order, rotation=45, ha="right")
        ax.set_title(
            game_ordering.get(task, task)
        )  # Use display name from game_ordering
        ax.set_ylabel("Progress (%)")
        ax.set_ylim(0, 100)  # Set y-axis limits to 0-100
        ax.grid(axis="y", linestyle="--", linewidth=0.7)
        if ax == axes[0]:
            ax.legend()

    # Hide any unused subplots
    for i in range(len(tasks), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# Plot and save the combined results
plot_combined_results(
    results, "LLM and VLM Results", "llm_vlm_results.png", game_ordering, model_colors
)
