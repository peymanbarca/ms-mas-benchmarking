import json
import matplotlib.pyplot as plt
import numpy as np

INPUT_JSON = "experiment4_results.json"

def load_data():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    return data


def plot_latency_vs_consistency(batch_metrics):
    """
    Scatter plot of: Success Rate (consistency) vs Average Latency.
    For Exp4 this is a single point, but the design allows multi-experiment comparison.
    """
    success = batch_metrics["success_rate"]
    latency = batch_metrics["avg_latency"]

    plt.figure(figsize=(7, 5))
    plt.scatter(success, latency, s=180)

    plt.xlabel("Consistency (Success Rate)")
    plt.ylabel("Average Total Latency (sec)")
    plt.title("Experiment 4 – Consistency vs Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_latency_distribution(results):
    """
    Per-trial latency distribution histogram.
    """
    latencies = [r["total_latency"] for r in results if "total_latency" in r]

    plt.figure(figsize=(8, 5))
    plt.hist(latencies, bins=15)
    plt.xlabel("Latency (sec)")
    plt.ylabel("Frequency")
    plt.title("Latency Distribution Across Trials")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_llm_vs_tool_latencies(results):
    """
    Bar chart: Avg LLM reasoning time vs Avg tool call latency.
    Supports multiple tools (inventory + order).
    """
    llm_times = []
    tool_times = []

    for r in results:
        if r.get("llm_reason_time"):
            llm_times.append(r["llm_reason_time"])
        if r.get("tool_latencies"):
            tool_times.extend(r["tool_latencies"])

    avg_llm = np.mean(llm_times) if llm_times else 0
    avg_tool = np.mean(tool_times) if tool_times else 0

    categories = ["LLM Reasoning", "Tool Calls"]
    values = [avg_llm, avg_tool]

    plt.figure(figsize=(7, 5))
    plt.bar(categories, values)
    plt.ylabel("Average Time (sec)")
    plt.title("Average LLM Reasoning vs Tool Call Latency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data()
    batch = data["batch_metrics"]
    results = data["results"]

    print("Loaded batch metrics:")
    print(json.dumps(batch, indent=4))

    # PLOTS
    plot_latency_vs_consistency(batch)
    plot_latency_distribution(results)
    plot_llm_vs_tool_latencies(results)
