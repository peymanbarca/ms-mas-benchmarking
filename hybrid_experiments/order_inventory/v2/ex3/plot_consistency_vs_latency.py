import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd

# ---------------- CONFIG ----------------
RESULTS_JSON = "exp3_results.json"  # results from exp3_runner.py
MONGO_URI = "mongodb://user:pass1@localhost:27017/"
DB_NAME = "retail_exp3"
ORDER_COLLECTION = "orders"
INVENTORY_COLLECTION = "inventory"
ITEM = "laptop"
INIT_STOCK = 10
NUM_THREADS = None

# ---------------- LOAD RESULTS ----------------
with open(RESULTS_JSON, "r") as f:
    results = json.load(f)

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
orders_col = db[ORDER_COLLECTION]
inventory_col = db[INVENTORY_COLLECTION]

# ---------------- COMPUTE CONSISTENCY ERROR ----------------
trial_ids = []
latencies = []
consistency_errors = []

for trial in results:
    trial_id = trial.get("trial")
    elapsed = trial.get("elapsed", 0)
    NUM_THREADS = trial.get("threads")

    # Calculate actual stock from inventory service
    stock_doc = inventory_col.find_one({"item": ITEM})
    actual_stock = stock_doc["stock"] if stock_doc else 0

    # Expected stock = INIT_STOCK - sum of reserved orders
    reserved_orders = orders_col.count_documents({"status": "reserved"})
    expected_stock = max(0, INIT_STOCK - reserved_orders * 2)  # qty=2 per order

    consistency_error = abs(expected_stock - actual_stock)

    trial_ids.append(trial_id)
    latencies.append(elapsed)
    consistency_errors.append(consistency_error)

# ---------------- CREATE DATAFRAME ----------------
df = pd.DataFrame({
    "trial": trial_ids,
    "latency": latencies,
    "consistency_error": consistency_errors
})

# ---------------- PLOT ----------------
plt.figure(figsize=(8, 6))
plt.scatter(df["latency"], df["consistency_error"], color="red", alpha=0.7)
plt.xlabel("Latency (s)")
plt.ylabel("Consistency Error (abs units)")
plt.title(f"Experiment 3: Consistency Error vs Latency, Num Threads : {int(NUM_THREADS)}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"exp3_latency_vs_consistency_num_threads_{int(NUM_THREADS)}.png", dpi=300)
plt.show()

# Optional: Aggregate by latency bins
bins = pd.cut(df["latency"], 10)
agg = df.groupby(bins)["consistency_error"].mean()
print("Average consistency error per latency bin:")
print(agg)
