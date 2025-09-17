import requests
import threading
import time
from tqdm import tqdm
import uuid
from pymongo import MongoClient
import psutil
import os

BASE_URL = "http://localhost:8081/order"
process = psutil.Process(os.getpid())

# MongoDB setup
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]
inventory = db["inventory"]
orders = db["orders"]


def reset_db():
    inventory.delete_many({})
    orders.delete_many({})


def place_order(item, qty, results, idx, delay):
    try:
        cpu_start = process.cpu_times()

        t1 = time.time()
        r = requests.post(BASE_URL, json={"item": item, "qty": qty, "delay_ms": delay*1000})
        res = r.json()
        results[idx] = res
        t2 = time.time()

        cpu_end = process.cpu_times()
        cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

        payload_req_bytes = res["payload"]["req_bytes"]
        payload_res_bytes = res["payload"]["res_bytes"]
        payload_size = payload_req_bytes + payload_res_bytes

        print(f'Total Response Took: {round((t2-t1), 3)}')
        with open(report_file_name, 'a') as f1:
            f1.write(f'Delay : {delay}, Total Response Took: {round((t2-t1), 3)},'
                     f' Status: {res["final_status"]}'
                     f' Payload: {payload_size} bytes, '
                     f' CPU time: {round(cpu_used, 5)} \n')

    except Exception as e:
        results[idx] = {"error": str(e)}


def run_experiment_parallel_order(trials=10, concurrent_orders=100):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    reset_db()
    input('Check DB state is clean, press any key to continue ...')

    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        results = {}
        threads = []

        # init stock in inventory for random item
        random_item_name = str(uuid.uuid4())[:5]
        inventory.insert_one({"item": random_item_name, "stock": INIT_STOCK})

        # Fire concurrent orders
        for i in range(concurrent_orders):
            th = threading.Thread(target=place_order, args=(random_item_name, 1, results, i, delay))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        # Check DB for consistency
        try:
            stock_left = db.inventory.find_one({"item": random_item_name})['stock']
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")
            print(f'Stock Left: {stock_left}, Total Completed Orders: {total_completed_orders}')

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == INIT_STOCK:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(e)
            failure_count += 1

    return success_count, failure_count


def run_experiment_sequential_order(trials=10, total_orders=100):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    reset_db()
    input('Check DB state is clean, press any key to continue ...')

    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        results = {}

        random_item_name = str(uuid.uuid4())[:5]
        inventory.insert_one({"item": random_item_name, "stock": INIT_STOCK})

        # Fire sequential orders
        for i in tqdm(range(total_orders)):
            place_order(item=random_item_name, qty=1, results=results, idx=i, delay=delay)

        # Check DB for consistency
        try:
            stock_left = db.inventory.find_one({"item": random_item_name})['stock']
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")
            print(f'Stock Left: {stock_left}, Total Completed Orders: {total_completed_orders}')

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == INIT_STOCK:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(e)
            failure_count += 1

    return success_count, failure_count


if __name__ == "__main__":
    delay = 0
    INIT_STOCK = 10
    # report_file_name = 'ms_sc3_seq.txt'
    report_file_name = 'ms_sc3_parallel.txt'
    with open(report_file_name, 'w') as f:
        f.write('')

    # success, failure = run_experiment_sequential_order()
    success, failure = run_experiment_parallel_order()

    print("Success:", success)
    print("Failure:", failure)
    print("Success rate:", success / (success + failure))

    with open(report_file_name, 'a') as f:
        f.write(f'\n\n Success: {success}, Failure: {failure}, Success rate: {success / (success + failure)}')
