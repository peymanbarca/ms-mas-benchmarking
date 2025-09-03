import requests
import threading
import time
import random
from pymongo import MongoClient

# Config
N_TRIALS = 20
API_URL = "http://localhost:8081/order"  # OrderService endpoint
ITEM = "item123"
INIT_STOCK = 10

# MongoDB setup
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]
inventory = db["inventory"]
orders = db["orders"]


def reset_db():
    inventory.delete_many({})
    orders.delete_many({})
    inventory.insert_one({"item": ITEM, "stock": INIT_STOCK})


def place_order(trial_id):
    qty = 1
    try:
        res = requests.post(API_URL, json={"item": ITEM, "qty": qty}, timeout=10)
        status = res.status_code
        if status == 200:
            print(f"[Trial {trial_id}] Reservation Request Succeeded")
        else:
            print(f"[Trial {trial_id}] Request failed: {status}")
    except Exception as e:
        print(f"[Trial {trial_id}] Request failed: {e}")


def run_experiment_parallel_order():
    reset_db()
    input('Check DB state is clean, press any key to continue ...')

    threads = []

    # Launch parallel order requests
    for i in range(N_TRIALS):
        t = threading.Thread(target=place_order, args=(i,))
        t.start()
        threads.append(t)
        # time.sleep(0.05)

    # Wait for all to complete
    for t in threads:
        t.join()

    # Gather results
    reserved = orders.count_documents({"status": "reserved"})
    out_of_stock = orders.count_documents({"status": "out_of_stock"})
    init_pending = orders.count_documents({"status": "INIT"})  # never updated
    stock = inventory.find_one({"item": ITEM})["stock"]

    print("\n=== Trial Summary before ===")
    print(f"Initial stock: {INIT_STOCK}")
    print(f"Orders attempted: {N_TRIALS}")

    print(f"Reserved: {reserved}")
    print(f"Out Of Stock: {out_of_stock}")
    print(f"Init pending: {init_pending}")
    print(f"Final stock: {stock}")

    time.sleep(15)

    # Gather results
    reserved = orders.count_documents({"status": "reserved"})
    out_of_stock = orders.count_documents({"status": "out_of_stock"})
    init_pending = orders.count_documents({"status": "INIT"})  # never updated
    stock = inventory.find_one({"item": ITEM})["stock"]

    print("\n=== Trial Summary after ===")
    print(f"Reserved: {reserved}")
    print(f"Out Of Stock: {out_of_stock}")
    print(f"Init pending: {init_pending}")
    print(f"Final stock: {stock}")

    print(f"Over-reservation? {reserved > INIT_STOCK}")
    print("=====================")


if __name__ == "__main__":
    run_experiment_parallel_order()
