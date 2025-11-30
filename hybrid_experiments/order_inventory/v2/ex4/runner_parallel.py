import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_orchestrator import invoke_agent   # your Exp4 agent

N_TRIALS = 50
MAX_WORKERS = 10
ITEM = "laptop"
QTY = 2

REPORT_FILE = "experiment4_results.json"

def run_single_trial(trial_id: int):
    """
    Runs one orchestration request through the LLM agent orchestrator.
    Returns a dict that matches plotting requirements.
    """
    start = time.time()
    try:
        result = invoke_agent(
            {
                "item": ITEM,
                "qty": QTY,
                "trial": trial_id
            }
        )
        latency = time.time() - start

        result_record = {
            "trial": trial_id,
            "success": result.get("consistency_ok", False),
            "total_latency": result.get("total_latency", latency),
            "llm_reason_time": result.get("llm_reason_time", None),
            "tool_latencies": result.get("tool_latencies", []),
            "final_state": result.get("final_state", {})
        }
        print(f"[Trial {trial_id}] SUCCESS: latency={result_record['total_latency']:.3f}s")
        return result_record

    except Exception as e:
        latency = time.time() - start
        print(f"[Trial {trial_id}] ERROR: {e}")
        return {
            "trial": trial_id,
            "success": False,
            "total_latency": latency,
            "error": str(e)
        }


def run_parallel_experiments():
    """
    Run N trials in parallel and aggregate metrics for plotting.
    """
    print(f"Running {N_TRIALS} trials with max {MAX_WORKERS} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_trial, i) for i in range(1, N_TRIALS + 1)]

        for f in as_completed(futures):
            results.append(f.result())

    # Aggregate metrics
    latencies = [r["total_latency"] for r in results if "total_latency" in r]
    successes = [r for r in results if r.get("success", False)]

    batch_metrics = {
        "num_trials": N_TRIALS,
        "num_success": len(successes),
        "success_rate": len(successes) / N_TRIALS if N_TRIALS > 0 else 0.0,
        "avg_latency": statistics.mean(latencies) if latencies else None,
        "median_latency": statistics.median(latencies) if latencies else None,
        "variance_latency": statistics.variance(latencies) if len(latencies) > 1 else 0.0,
    }

    print("\n=== Batch Summary ===")
    print(json.dumps(batch_metrics, indent=4))

    # Write logs for plotting scripts
    output = {
        "batch_metrics": batch_metrics,
        "results": results
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved results to {REPORT_FILE}")


if __name__ == "__main__":
    run_parallel_experiments()
