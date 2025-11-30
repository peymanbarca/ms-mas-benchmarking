import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load JSON results
# -----------------------
RESULT_FILE = "experiment5_graph_results.json"

with open(RESULT_FILE, "r") as f:
    data = json.load(f)

# -----------------------
# Compute latency per trial
# -----------------------
latencies = []
consistency_flags = []

for trial in data:
    # Compute latency from elapsed timestamps if available
    elapsed = trial.get("elapsed", 0)
    latencies.append(elapsed)

    # Consistency check: order status matches reservation in trace
    trace = trial.get("trace", [])
    reserved = any(step.get("step")=="reserve_inventory" and step.get("out", {}).get("status")=="reserved" for step in trace)
    order_status = any(step.get("step")=="update_order_status" and step.get("out", {}).get("status")=="reserved" for step in trace)
    consistent = int(reserved and order_status)
    consistency_flags.append(consistent)

# -----------------------
# Aggregate: compute consistency rate per latency bucket
# -----------------------
latencies = np.array(latencies)
consistency_flags = np.array(consistency_flags)

# For plotting, group latencies into bins
bins = np.linspace(latencies.min(), latencies.max(), 10)
bin_indices = np.digitize(latencies, bins)

avg_latency_per_bin = []
consistency_rate_per_bin = []

for i in range(1, len(bins)+1):
    indices = np.where(bin_indices == i)[0]
    if len(indices) > 0:
        avg_latency_per_bin.append(latencies[indices].mean())
        consistency_rate_per_bin.append(consistency_flags[indices].mean())

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,6))
plt.plot(avg_latency_per_bin, consistency_rate_per_bin, marker='o', linestyle='-', color='blue')
plt.xlabel("Average Latency (s)")
plt.ylabel("Consistency Rate")
plt.title("Experiment 5: Latency vs Consistency Tradeoff")
plt.grid(True)
plt.ylim(0, 1.05)
plt.show()
