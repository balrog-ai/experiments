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
}

include_models = list(model_colors.keys())

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

# Compute average performance for each model in each modality
average_performance = {"llm": {}, "vlm": {}}

for modality in ["llm", "vlm"]:
    for model in include_models:
        total_value = 0.0
        total_error = 0.0
        total_count = 0
        for game in results[modality]:
            if model in results[modality][game]:
                value, error = results[modality][game][model]
                total_value += value
                total_error += error**2  # Sum of squared errors
                total_count += 1
        if total_count > 0:
            avg_value = total_value / total_count
            avg_error = (total_error**0.5) / total_count  # Combined standard error
            average_performance[modality][model] = (avg_value, avg_error)
        else:
            # Model has no data in this modality
            average_performance[modality][model] = (None, None)

# Compute overall average performance for each model across both modalities
model_overall_performance = {}

for model in include_models:
    modality_avgs = []
    for modality in ["llm", "vlm"]:
        avg_value, avg_error = average_performance[modality].get(model, (None, None))
        if avg_value is not None:
            modality_avgs.append(avg_value)
    if modality_avgs:
        overall_avg = sum(modality_avgs) / len(modality_avgs)
        model_overall_performance[model] = overall_avg
    else:
        model_overall_performance[model] = None  # No data for this model


# Sort models based on overall average performance (highest first)
def get_sort_key(model):
    avg = model_overall_performance.get(model)
    if avg is None:
        return -float("inf")  # Ensure models with no data are at the bottom
    else:
        return avg


include_models_sorted = sorted(include_models, key=get_sort_key, reverse=True)


# Function to generate LaTeX table for average performance
def generate_average_performance_table(average_performance, title):
    latex_table = f"\\begin{{table}}[h]\n\\centering\n\\caption{{{title}}}\n"
    latex_table += "\\begin{tabular}{lcc}\n\\hline\n"
    latex_table += (
        "Model & LLM Avg Progress (\\%) & VLM Avg Progress (\\%) \\\\\n\\hline\n"
    )
    for model in include_models_sorted:
        llm_avg, llm_err = average_performance["llm"].get(model, (None, None))
        vlm_avg, vlm_err = average_performance["vlm"].get(model, (None, None))
        llm_str = (
            f"{llm_avg:.2f} $\\pm$ {llm_err:.2f}" if llm_avg is not None else "N/A"
        )
        vlm_str = (
            f"{vlm_avg:.2f} $\\pm$ {vlm_err:.2f}" if vlm_avg is not None else "N/A"
        )
        latex_table += f"{model} & {llm_str} & {vlm_str} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex_table


# Generate and print LaTeX table for average performance
average_table = generate_average_performance_table(
    average_performance, "Average Model Performance"
)
print("LaTeX Table for Average Model Performance:")
print(average_table)


# Function to generate LaTeX table for per-game performance
def generate_per_game_table(game, results, title):
    latex_table = f"\\begin{{table}}[h]\n\\centering\n\\caption{{{title}}}\n"
    latex_table += "\\begin{tabular}{lcc}\n\\hline\n"
    latex_table += "Model & LLM Progress (\\%) & VLM Progress (\\%) \\\\\n\\hline\n"
    for model in include_models_sorted:
        llm_value, llm_error = results["llm"].get(game, {}).get(model, (None, None))
        vlm_value, vlm_error = results["vlm"].get(game, {}).get(model, (None, None))
        llm_str = (
            f"{llm_value:.2f} $\\pm$ {llm_error:.2f}"
            if llm_value is not None
            else "N/A"
        )
        vlm_str = (
            f"{vlm_value:.2f} $\\pm$ {vlm_error:.2f}"
            if vlm_value is not None
            else "N/A"
        )
        latex_table += f"{model} & {llm_str} & {vlm_str} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex_table


# Generate and print LaTeX tables for each game
for game in sorted(game_names_set):
    title = f"Performance on {game}"
    game_table = generate_per_game_table(game, results, title)
    print(f"LaTeX Table for {title}:")
    print(game_table)
