import json
import os
from jinja2 import Environment, FileSystemLoader

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader("."))

# Load the template
template_index = env.get_template("template_index.html")
template_viewer = env.get_template("template_viewer.html")

# Load JSON data
with open("data.json", "r") as f:
    data = json.load(f)

# For each model, read the README.md content and add it to the data
for leaderboard in data["leaderboards"]:
    for item in leaderboard["results"]:
        readme_path = os.path.join(
            "..", item["folder"], "README.md"
        )  # Adjust the path as needed
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
        except FileNotFoundError:
            readme_content = "README.md not provided."

        # Include the README content in the model's data
        item["readme_content"] = readme_content

# Render the templates with the updated data
output_html_index = template_index.render(data)
output_html_viewer = template_viewer.render(data=data)  # Pass 'data' explicitly

# Write the output to HTML files
with open("../index.html", "w", encoding="utf-8") as f:
    f.write(output_html_index)

with open("../viewer.html", "w", encoding="utf-8") as f:
    f.write(output_html_viewer)

print("DONE")
