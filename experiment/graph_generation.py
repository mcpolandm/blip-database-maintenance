import json
from os import path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

file_path = Path("data/analysis/summary.json")

with file_path.open("r", encoding="utf-8") as f:
    file_results = json.load(f)

policies= ["distance", "provenance", "answer-aware provenance", "always recompute", "text_diff"]
time_b = {"Recompute Answer": [], "Compute Provenance": []}
llm_calls_b = {"Recompute Answer": [], "Compute Provenance": []}
tokens_b = {"Recompute Answer": [], "Compute Provenance": []}
accuracy_b = []
time_e = {"Recompute Answer": [], "Compute Provenance": []}
llm_calls_e = {"Recompute Answer": [], "Compute Provenance": []}
tokens_e = {"Recompute Answer": [], "Compute Provenance": []}
accuracy_e = []
for data in file_results:
    if data["task_type"] == "boolean":
        accuracy_b.append(data["accuracy"])
        llm_calls_b["Recompute Answer"].append(data["mean_incremental_llm_calls"])
        llm_calls_b["Compute Provenance"].append(data["mean_provenance_calls"])
        time_b["Recompute Answer"].append(data["mean_policy_decision_time_seconds"])
        time_b["Compute Provenance"].append(data["mean_provenance_time_seconds"])
        tokens_b["Recompute Answer"].append(data["mean_incremental_tokens"])
        tokens_b["Compute Provenance"].append(data["mean_provenance_tokens"])
    elif data["task_type"] == "extraction":
        accuracy_e.append(data["accuracy"])
        llm_calls_e["Recompute Answer"].append(data["mean_incremental_llm_calls"])
        llm_calls_e["Compute Provenance"].append(data["mean_provenance_calls"])
        time_e["Recompute Answer"].append(data["mean_policy_decision_time_seconds"])
        time_e["Compute Provenance"].append(data["mean_provenance_time_seconds"])
        tokens_e["Recompute Answer"].append(data["mean_incremental_tokens"])
        tokens_e["Compute Provenance"].append(data["mean_provenance_tokens"])

for i, policy in enumerate(policies):
    print(f"{policy}: accuracy {accuracy_b[i]}, llm_calls {llm_calls_b['Recompute Answer'][i]}")

width = 0.5
fig, ax = plt.subplots()
bottom = np.zeros(5)

for boolean, weight_count in llm_calls_b.items():
    p = ax.bar(policies, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.set_title("LLM Calls Across Policies")
ax.set_xlabel("Policies")
ax.set_ylabel("Mean LLM Calls")
ax.legend(loc="upper right")
plt.xticks(rotation=45, ha='right')  # Aligns the end of the text with the tick mark
plt.tight_layout() 
plt.show()