import requests
import threading
import time
from tqdm import tqdm
import uuid
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8081/order"


process = psutil.Process(os.getpid())


def place_order(item, qty, results, idx, delay):
    try:
        cpu_start = process.cpu_times()

        t1 = time.time()
        r = requests.post(BASE_URL, json={"item": item, "qty": qty, "delay": delay})
        res = r.json()
        results[idx] = res
        t2 = time.time()

        cpu_end = process.cpu_times()
        cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

        req_size = len(r.request.body or b"") + sum(len(str(v)) for v in r.request.headers.values())
        res_size = len(r.content) + sum(len(str(v)) for v in r.headers.values())
        payload_size = req_size + res_size

        # print(f'Total Response Took: {round((t2-t1), 3)}')
        with open(report_file_name, 'a') as f1:
            f1.write(f'Delay : {delay}, '
                     f'Total Response Took: {round((t2-t1), 3)}, '
                     f'Status: {res["final_status"]}, '
                     f'Payload: {payload_size} bytes, '
                     f'CPU time: {round(cpu_used, 5)} \n')

    except Exception as e:
        results[idx] = {"error": str(e)}


def run_experiment_parallel_order(total_orders=100):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    requests.post("http://localhost:8081/clear_orders", json={})
    requests.post("http://localhost:8082/clear_stocks", json={})
    input('Check DB state is clean, press any key to continue ...')

    for t in range(n_trials):
        print(f"\nTrial {t + 1}/{n_trials}")
        results = {}
        random_item_name = str(uuid.uuid4())[:5]

        # initialize stock
        requests.post("http://localhost:8082/init_stock", json={"item": random_item_name})

        # thread pool size
        pool_size = int(total_orders / init_stock) + 2 * delay
        print(f"Using thread pool of size: {pool_size}")

        # submit tasks to thread pool
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(place_order, random_item_name, qty, results, i, delay)
                for i in range(total_orders)
            ]

            # wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Order thread error: {e}")

        # validate consistency
        try:
            inv = requests.get(f"http://localhost:8082/debug_stock?item={random_item_name}").json()
            stock_left = inv["stock"]
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")

            print(f"Stock Left: {stock_left}, Completed Orders: {total_completed_orders}")

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == int(init_stock / qty):
                success_count += 1
                print('success')
            else:
                failure_count += 1
                print('failure')
        except Exception as e:
            print(f"Validation error: {e}")
            failure_count += 1

    return success_count, failure_count


def run_experiment_sequential_order(total_orders=100):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    requests.post(f"http://localhost:8081/clear_orders", json={})
    requests.post(f"http://localhost:8082/clear_stocks", json={})
    input('Check DB state is clean, press any key to continue ...')

    for t in range(n_trials):
        print(f"Trial {t+1}/{n_trials}")
        results = {}
        random_item_name = str(uuid.uuid4())[:5]

        # init stock in inventory for random item
        requests.post(f"http://localhost:8082/init_stock", json={"item": random_item_name})

        # Fire sequential orders
        for i in tqdm(range(total_orders)):
            place_order(item=random_item_name, qty=qty, results=results, idx=i, delay=delay)

        # Check DB for consistency
        try:
            inv = requests.get("http://localhost:8082/debug_stock?item=a").json()
            stock_left = inv["stock"]
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")
            print(f'Stock Left: {stock_left}, Total Completed Orders: {total_completed_orders}')

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == int(init_stock / qty):
                success_count += 1
                print('success')
            else:
                failure_count += 1
                print('failure')
        except Exception as e:
            print(e)
            failure_count += 1

    return success_count, failure_count


if __name__ == "__main__":
    delay = 2
    # report_file_name = 'ms_sc1_sequential.txt'
    report_file_name = 'ms_sc1_parallel.txt'
    init_stock = 10
    n_trials = 5
    qty = 2

    with open(report_file_name, 'w') as f:
        f.write('')

    # success, failure = run_experiment_sequential_order()
    success, failure = run_experiment_parallel_order()

    print("Success:", success)
    print("Failure:", failure)
    print("Success rate:", success / (success + failure))

    with open(report_file_name, 'a') as f:
        f.write(f'\n\n Success: {success}, Failure: {failure}, Success rate: {success / (success + failure)}')
