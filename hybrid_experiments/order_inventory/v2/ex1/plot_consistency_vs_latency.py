import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Load experiment results
# --------------------------------------------------------------
with open("result/exp1_results.json", "r") as f:
    trials = json.load(f)


latencies = []
consistency_errors = []

# --------------------------------------------------------------
# Compute metrics for each trial
# --------------------------------------------------------------
for t in trials:
    response_time = t["response_time"]
    latencies.append(response_time)

    errors = t.get("consistency_errors", [])
    if errors:
        consistency_errors.append(max(errors))  # worst violation for the trial
    else:
        consistency_errors.append(0)

latencies = np.array(latencies)
consistency_errors = np.array(consistency_errors)

# --------------------------------------------------------------
# Plot 1: Consistency Error vs Latency
# --------------------------------------------------------------
plt.figure()
plt.scatter(latencies, consistency_errors)
plt.xlabel("Latency (seconds)")
plt.ylabel("Consistency Error")
plt.title("Consistency Error vs Latency")
plt.grid(True)
plt.tight_layout()
plt.savefig("consistency_vs_latency.png")

# --------------------------------------------------------------
# Plot 2: Latency Distribution (Histogram)
# --------------------------------------------------------------
plt.figure()
plt.hist(latencies, bins=20)
plt.xlabel("Latency (seconds)")
plt.ylabel("Frequency")
plt.title("Latency Distribution Across Trials")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_distribution.png")

print("Plots saved as:")
print(" - consistency_vs_latency.png")
print(" - latency_distribution.png")
