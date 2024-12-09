import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_correlation_matrices(data):
    leaderboards = data["leaderboards"]
    correlation_matrices = {}

    # Define the list of environments to consider
    environments = ["babyai", "babaisai", "crafter", "textworld", "minihack", "nle"]

    for leaderboard in leaderboards:
        lb_name = leaderboard["name"]
        results = leaderboard["results"]

        # List to hold each agent's performance
        agent_performances = []

        for agent in results:
            agent_name = agent["name"]
            performance = {"agent": agent_name}

            # Extract performance scores for each environment
            for env in environments:
                if env in agent:
                    # Use the mean score (first element of the list)
                    performance[env] = agent[env][0]

            agent_performances.append(performance)

        # Create a DataFrame from the agent performances
        df = pd.DataFrame(agent_performances)

        # Check if there are at least two agents to compute correlation
        if len(df) < 2:
            print(f"Not enough data to compute correlation matrix for {lb_name}.")
            continue

        # Set 'agent' as the index
        df.set_index("agent", inplace=True)

        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Store the correlation matrix
        correlation_matrices[lb_name] = corr_matrix

    return correlation_matrices


# Load the data from 'data.json'
with open("template/data.json", "r") as f:
    data = json.load(f)

# Create the correlation matrices
corr_matrices = create_correlation_matrices(data)

# Plot the correlation matrices and find the environment with the highest average correlation
for lb_name in ["LLM", "VLM"]:
    if lb_name in corr_matrices:
        corr_matrix = corr_matrices[lb_name]

        # Plot the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Matrix for {lb_name}")
        plt.tight_layout()
        plt.show()

        # Calculate the average correlation per environment, excluding self-correlation
        avg_corr = corr_matrix.apply(lambda x: x.drop(labels=x.name).mean(), axis=1)

        # Find the environment with the highest average correlation
        max_env = avg_corr.idxmax()
        print(
            f"The environment with the highest average correlation in {lb_name} is '{max_env}' with an average correlation of {avg_corr[max_env]:.2f}"
        )
    else:
        print(f"No correlation matrix available for {lb_name}.")
