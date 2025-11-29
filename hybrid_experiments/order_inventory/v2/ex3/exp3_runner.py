import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient

# ---------------- CONFIG ----------------
ORDER_SERVICE_URL = "http://127.0.0.1:8000/create_order"
ITEM = "laptop"
INIT_STOCK = 10
QTY = 2
N_TRIALS = 50
REPORT_FILE = "result/exp3_results.json"
MAX_WORKERS = N_TRIALS / 1  # Number of concurrent threads

requests.post("http://localhost:8000/clear_orders", json={})
requests.post("http://localhost:8001/reset_stocks", json={"item": ITEM, "stock": INIT_STOCK})
input('Check DB state is clean, press any key to continue ...')

results = []

def run_trial(trial_id: int):
    payload = {"item": ITEM, "qty": QTY}
    start = time.time()
    try:
        resp = requests.post(ORDER_SERVICE_URL, json=payload, timeout=5)
        elapsed = time.time() - start
        if resp.status_code == 200:
            result = resp.json()
            result["trial"] = trial_id
            result["elapsed"] = round(elapsed, 3)
            result["threads"] = MAX_WORKERS
            print(f"Trial {trial_id}: {result}")
            return result
        else:
            print(f"Trial {trial_id}: ERROR")
            return {"trial": trial_id, "status": "error", "elapsed": round(elapsed,3)}
    except Exception as e:
        elapsed = time.time() - start
        print(f"Trial {trial_id}: Exception {e}")
        return {"trial": trial_id, "status": "error", "elapsed": round(elapsed,3)}

# ---------------- PARALLEL EXECUTION ----------------
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(run_trial, i) for i in range(1, N_TRIALS + 1)]
    for future in as_completed(futures):
        results.append(future.result())

# Save all results
with open(REPORT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nAll {N_TRIALS} trials completed. Results saved in {REPORT_FILE}")
