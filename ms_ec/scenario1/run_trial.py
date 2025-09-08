import requests
import threading
import time
from tqdm import tqdm
import uuid

BASE_URL = "http://localhost:8081/order"


def place_order(item, qty, results, idx, delay):
    try:
        t1 = time.time()
        r = requests.post(BASE_URL, json={"item": item, "qty": qty, "delay": delay})
        results[idx] = r.json()
        t2 = time.time()
        print(f'Reservation Response Took: {round((t2-t1), 3)}')
    except Exception as e:
        results[idx] = {"error": str(e)}


def run_experiment_parallel_order(trials=5, concurrent_orders=20):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    requests.post(f"http://localhost:8081/clear_orders", json={})
    requests.post(f"http://localhost:8082/clear_stocks", json={})

    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        results = {}
        threads = []
        random_item_name = str(uuid.uuid4())[:5]

        # init stock in inventory for random item
        requests.post(f"http://localhost:8082/init_stock", json={"item": random_item_name})

        # Fire concurrent orders
        delay = 3
        for i in range(concurrent_orders):
            th = threading.Thread(target=place_order, args=(random_item_name, 1, results, i, delay))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        # Check DB for consistency
        try:
            inv = requests.get(f"http://localhost:8082/debug_stock?item={random_item_name}").json()
            stock_left = inv["stock"]
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")
            print(f'Stock Left: {stock_left}, Total Completed Orders: {total_completed_orders}')

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == 10:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(e)
            failure_count += 1

    return success_count, failure_count


def run_experiment_sequential_order(trials=5, total_orders=20):
    success_count = 0
    failure_count = 0

    # clear previous orders/stocks for clean trial run
    requests.post(f"http://localhost:8081/clear_orders", json={})
    requests.post(f"http://localhost:8082/clear_stocks", json={})

    for t in range(trials):
        print(f"Trial {t+1}/{trials}")
        results = {}
        random_item_name = str(uuid.uuid4())[:5]

        # init stock in inventory for random item
        requests.post(f"http://localhost:8082/init_stock", json={"item": random_item_name})

        # Fire sequential orders
        for i in tqdm(range(total_orders)):
            place_order(item=random_item_name, qty=1, results=results, idx=i)

        # Check DB for consistency
        try:
            inv = requests.get("http://localhost:8082/debug_stock?item=a").json()
            stock_left = inv["stock"]
            total_completed_orders = sum(1 for r in results.values() if r.get("final_status") == "COMPLETED")
            print(f'Stock Left: {stock_left}, Total Completed Orders: {total_completed_orders}')

            if stock_left >= 0 and stock_left + \
                    total_completed_orders == 10:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(e)
            failure_count += 1

    return success_count, failure_count


if __name__ == "__main__":
    # success, failure = run_experiment_sequential_order()
    success, failure = run_experiment_parallel_order()

    print("Success:", success)
    print("Failure:", failure)
    print("Success rate:", success / (success + failure))
