import requests
import threading
import time
import random
from pymongo import MongoClient
import uuid

# MongoDB setup
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]
inventory = db["inventory"]
orders = db["orders"]


def reset_db():
    inventory.delete_many({})
    orders.delete_many({})


def place_order(idx, item, delay):
    qty = 1
    try:
        t1 = time.time()
        r = requests.post(API_URL, json={"item": item, "qty": qty, "delay": delay}, timeout=10)
        status = r.status_code
        t2 = time.time()
        if status == 200:
            print(f"[Trial {idx}] Reservation Request Succeeded")
        else:
            print(f"[Trial {idx}] Request failed: {status}")

        with open(report_file_name, 'a') as f1:
            f1.write(
                f'Delay : {delay}, Total Response Took: {round((t2 - t1), 3)} \n')
    except Exception as e:
        print(f"[Trial {idx}] Request failed: {e}")


def run_experiment_parallel_order():
    reset_db()
    input('Check DB state is clean, press any key to continue ...')

    for trial in range(N_TRIALS):
        with open(report_file_name, 'a') as f1:
            f1.write(f'---- Trial {trial} ----\n\n')
        random_item_name = str(uuid.uuid4())[:5]
        inventory.insert_one({"item": random_item_name, "stock": INIT_STOCK})

        threads = []

        # Launch parallel order requests
        for i in range(ORDERS_WITHIN_EACH_TRIAL):
            t = threading.Thread(target=place_order, args=(i, random_item_name, delay))
            t.start()
            threads.append(t)
            # time.sleep(0.05)

        # Wait for all to complete
        for t in threads:
            t.join()

        # Gather results directly from DB
        reserved = orders.count_documents({"status": "reserved", "item": random_item_name})
        out_of_stock = orders.count_documents({"status": "out_of_stock", "item": random_item_name})
        init_pending = orders.count_documents({"status": "INIT", "item": random_item_name})  # never updated
        stock = inventory.find_one({"item": random_item_name})["stock"]

        with open(report_file_name, 'a') as f1:
            f1.write('----------------\n\n')
            f1.write(f'Initial stock: {INIT_STOCK},\nOrders attempted: {ORDERS_WITHIN_EACH_TRIAL}\n')
            f1.write("\n=== Trial Summary before ===\n")
            f1.write(f"Reserved orders: {reserved}\n")
            f1.write(f"Out Of Stock orders: {out_of_stock}\n")
            f1.write(f"Init pending orders: {init_pending}\n")
            f1.write(f"Final stock remaining: {stock}\n\n")

        time.sleep(3 * delay)

        # Gather results directly from DB again to see eventual consistency
        reserved = orders.count_documents({"status": "reserved", "item": random_item_name})
        out_of_stock = orders.count_documents({"status": "out_of_stock", "item": random_item_name})
        init_pending = orders.count_documents({"status": "INIT", "item": random_item_name})  # never updated
        stock = inventory.find_one({"item": random_item_name})["stock"]
        ec_met = out_of_stock + reserved == ORDERS_WITHIN_EACH_TRIAL and init_pending == 0 and stock >= 0

        with open(report_file_name, 'a') as f1:
            f1.write("\n=== Trial Summary after ===\n")
            f1.write(f"Reserved orders: {reserved}\n")
            f1.write(f"Out Of Stock orders: {out_of_stock}\n")
            f1.write(f"Init pending orders: {init_pending}\n")
            f1.write(f"Final stock remaining: {stock}\n")
            f1.write('-------------------')
            f1.write(f"Over-reservation occurred? {reserved > INIT_STOCK}\n")
            f1.write(f"Eventual consistency met? {ec_met}\n\n")


if __name__ == "__main__":
    # Config
    N_TRIALS = 5
    ORDERS_WITHIN_EACH_TRIAL = 100
    API_URL = "http://localhost:8081/order"  # OrderService endpoint
    INIT_STOCK = 10
    delay = 3
    report_file_name = 'ms_sc2_parallel.txt'
    with open(report_file_name, 'w') as f:
        f.write('')

    run_experiment_parallel_order()
