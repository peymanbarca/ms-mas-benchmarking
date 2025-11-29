import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

INIT_STOCK=10

# Config
RESULTS_JSON = "exp2_results.json"  # path to experiment results

# Load experiment results
with open(RESULTS_JSON, "r") as f:
    results = json.load(f)

# Prepare lists
trial_ids = []
latencies = []
consistency_errors = []

for trial in results:
    trial_ids.append(trial.get("trial"))
    elapsed = trial.get("elapsed", 0)
    final_state = trial.get("final_state", {})
    # compute consistency error = |expected - actual stock|
    remaining = final_state.get("remaining")
    qty = final_state.get("qty", 0)
    status = final_state.get("status", "")
    # Expected remaining stock assuming successful reservations
    committed = 0
    if status == "reserved":
        committed = 1
    elif status == "out_of_stock":
        committed = 0
    expected_remaining = max(0, INIT_STOCK - committed * qty)
    if remaining is None:
        remaining = expected_remaining  # fallback
    consistency_error = abs(expected_remaining - remaining)
    latencies.append(elapsed)
    consistency_errors.append(consistency_error)

# Create dataframe
df = pd.DataFrame({
    "trial": trial_ids,
    "latency": latencies,
    "consistency_error": consistency_errors
})

# Plot consistency_error vs latency
plt.figure(figsize=(8,6))
plt.scatter(df["latency"], df["consistency_error"], color="blue", alpha=0.7)
plt.xlabel("Latency (s)")
plt.ylabel("Consistency Error (abs units)")
plt.title("Consistency Error vs Latency per Trial (Experiment 2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_vs_consistency.png", dpi=300)
plt.show()

# Optional: aggregate stats per latency bin
bins = np.linspace(min(latencies), max(latencies), 10)
df["latency_bin"] = pd.cut(df["latency"], bins)
agg = df.groupby("latency_bin")["consistency_error"].mean()
print("Average consistency error per latency bin:")
print(agg)
