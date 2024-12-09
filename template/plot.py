import matplotlib.pyplot as plt
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    # "o1-preview": "000000",
}

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Function to plot average progression for LLM and VLM with hatch over colored bars
def plot_average_progression(results, title, filename, model_colors):
    include_models = list(model_colors.keys())

    # Initialize data storage for averages
    avg_progression = {"llm": {}, "vlm": {}}

    # Compute averages for LLM and VLM
    for modality in ["llm", "vlm"]:
        for model in include_models:
            values = []
            for task, task_data in results[modality].items():
                if model in task_data:
                    values.append(task_data[model][0])  # Extract progression value
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
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(title, fontsize=22)
    ax.set_ylabel("Average Progress (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7)
    ax.set_yticks(np.arange(0, 101, 10))  # Add lines for every 10% increment

    # Use the custom legend handles
    ax.legend(handles=[grey_patch, grey_hatched_patch])

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


import matplotlib.patches as mpatches


def plot_combined_results(
    results, title, filename, game_ordering, model_colors, tasks_to_plot=None
):
    include_models = list(model_colors.keys())
    # Filter tasks based on the specified tasks_to_plot and data availability
    if tasks_to_plot is None:
        tasks = [
            task
            for task in game_ordering.keys()
            if task in results["llm"] or task in results["vlm"]
        ]
    else:
        tasks = [
            task
            for task in tasks_to_plot
            if task in results["llm"] or task in results["vlm"]
        ]

    if len(tasks) == 0:
        print("No tasks to plot.")
        return

    if len(tasks) == 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        axes = [ax]
        fig.suptitle(title, fontsize=22)
    else:
        # Adjusted subplot arrangement for multiple tasks
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
            edgecolor="black",
            label="_nolegend_",  # Exclude these bars from the legend
        )
        # Plot vlm bars with hatching
        ax.bar(
            x + bar_width / 2,
            vlm_values,
            width=bar_width,
            yerr=vlm_errors,
            capsize=5,
            color=colors,
            edgecolor="black",
            hatch="//",
            label="_nolegend_",  # Exclude these bars from the legend
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

        # Create custom legend handles with grey color
        grey_patch = mpatches.Patch(facecolor="grey", edgecolor="black", label="LLM")
        grey_hatched_patch = mpatches.Patch(
            facecolor="grey", edgecolor="black", hatch="//", label="VLM"
        )

        # Add legend to the first axis
        if ax == axes[0]:
            ax.legend(handles=[grey_patch, grey_hatched_patch])

    # Hide any unused subplots (if any)
    if len(tasks) > 1:
        for i in range(len(tasks), len(axes)):
            fig.delaxes(axes[i])

    plt.tight_layout(pad=2)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# Plot only the NetHack game
plot_combined_results(
    results,
    title="NetHack Results",
    filename="nethack_results.png",
    game_ordering=game_ordering,
    model_colors=model_colors,
    tasks_to_plot=["nle"],
)
