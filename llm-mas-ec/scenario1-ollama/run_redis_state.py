import uuid
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional, Dict, List, Any
from pymongo import MongoClient
import json
import time
import re
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import redis

logging.basicConfig(
    filename='llm_mas_sc1_traces.log',  # Specify the log file name
    level=logging.DEBUG,          # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
)

process = psutil.Process(os.getpid())


class RedisStateStore:
    def __init__(self, host="localhost", port=6379, db=0):
        self.r = redis.Redis(host='', port=15385, db=0, password='')

    def save_state(self, key: str, state: dict):
        """Persist state as JSON under a given key"""
        self.r.set(key, json.dumps(state), ex = 300)

    def load_state(self, key: str) -> dict:
        """Fetch state JSON and return as dict"""
        data = self.r.get(key)
        return json.loads(data) if data else {}


# global redis store (could be injected instead)
state_store = RedisStateStore()


# 1. Define global state
# State just flows down the chain in a single process
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]  # INIT, reserved, out_of_stock


def real_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["retail_mas"]
    return client, db


def reset_db(item, init_stock):
    client, db = real_db()
    orders = db.orders
    inventory = db.inventory
    inventory.delete_many({})
    orders.delete_many({})
    stock = {"item": item, "stock": init_stock}
    inventory.insert_one(stock)


def get_final_stock(item):
    client, db = real_db()
    inventory = db.inventory
    final_stock = inventory.find_one({"item": item})
    return final_stock

# 2. Deterministic DB Agent
class DBAgent:
    def __init__(self, db, mode: str = 'MOCK'):
        self.mode = mode
        if mode == "REAL":
            if db is None:
                raise ValueError("MongoDB client must be provided in real mode")
            self.db = db
            self.orders = db.orders
            self.inventory = db.inventory

        elif mode == "MOCK":
            self.orders: List[Dict] = []
            self.inventory: List[Dict] = []
        else:
            raise ValueError("mode must be 'real' or 'mock'")

    def save_order(self, order_id: str, item: str, qty: int):
        if self.mode == "REAL":
            self.db.orders.insert_one({
                "_id": order_id,
                "item": item,
                "qty": qty,
                "status": "INIT"
            })
        else:  # mock mode
            self.orders.append({
                "_id": order_id,
                "item": item,
                "qty": qty,
                "status": "INIT"
            })

    def update_order(self, order_id: str, status: str):
        if self.mode == "REAL":
            self.db.orders.update_one({"_id": order_id}, {"$set": {"status": status}})
        else:
            for order in self.orders:
                if order["_id"] == order_id:
                    order["status"] = status
                    break

    def get_stock(self, item: str) -> int:
        if self.mode == "REAL":
            stock = self.db.inventory.find_one({"item": item})
            print(f"Stock in inventory DB is: {stock}")
            if not stock:
                stock = {"item": item, "stock": 10}
                self.db.inventory.insert_one(stock)
            return stock["stock"]
        else:
            stock = next((s for s in self.inventory if s["item"] == item), None)
            if not stock:
                stock = {"item": item, "stock": 10}
                self.inventory.append(stock)
            return stock["stock"]

    def update_stock(self, item: str, qty: int):
        if self.mode == "REAL":
            self.db.inventory.update_one(
                {"item": item},
                {"$inc": {"stock": -qty}},
                upsert=True
            )
        else:
            for s in self.inventory:
                if s["item"] == item:
                    s["stock"] -= qty
                    return


# 3. LLM setup
# llm = ChatOllama(model="tinyllama", temperature=0, base_url="http://127.0.0.1:11434")
llm = Ollama(model="qwen2")

order_prompt = """
You are the Order Agent.
Decide what to do with an incoming order.

Status: status_in

Rules:
- If Status is empty or INIT → create the order in DB and forward to inventory agent.
- If Status is reserved or out_of_stock → finalize order with given status and update DB.
Return only one JSON with keys: status(init/reserved/out_of_stock), forward (true/false).

"""

inventory_prompt = """
You are the Inventory Agent.
Decide whether to reserve stock.

Order ID: {order_id_in}
Item: {item_in}
Quantity: {qty_in}
Stock: {stock_in}

Rules:
- If Stock >= Quantity → reserved
- Otherwise → out_of_stock

Return JSON with keys: order_id, status.


"""


def parse_json_response(content: str, fallback: dict):
    """Extract JSON object from LLM output, safely parse it."""
    try:
        # Find the first {...} block in the text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return fallback
    except Exception as e:
        print("JSON parse error:", e)
        logging.error(f"JSON parse error: {e}")
        return fallback


# 4. Agent functions (sync now)
def order_agent(state: OrderState, db_ag: DBAgent):
    key = f"order:{state['order_id']}"
    state = state_store.load_state(key) or state
    print('value fetched from Redis: ', key, state)

    # First or second step, LLM decides what to do
    prompt = order_prompt.replace('status_in', state["status"] or "INIT")
    logging.debug(f"Order Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = llm.invoke(prompt)
    et = time.time()
    print(f"Order Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")
    logging.debug(f"Order Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")

    print(f"Order Agent: LLM Raw Response Content is \n {response} \n ...")
    logging.debug(f"Order Agent: LLM Raw Response Content is \n {response} \n ...")

    parsed = parse_json_response(
        response,
        fallback={
            "order_id": state["order_id"] or str(uuid.uuid4()),
            "status": "INIT",
            "forward": True,
        },
    )

    print(f"Order Agent: Requested Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")
    logging.debug(f"Order Agent: Requested Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")
    with open(report_file_name, "a") as f:
        f.write(f"\tOrder Agent Response: {parsed}\n")

    if str(parsed["status"]).lower() == "init":
        # Save INIT order
        db_ag.save_order(state["order_id"], state["item"], state["qty"])

    elif str(parsed["status"]).lower() in ["reserved", "out_of_stock"]:
        # Finalize order based on reservation response
        db_ag.update_order(state["order_id"], state["status"])

    state['status'] = parsed['status']
    state['forward'] = parsed['forward']
    state_store.save_state(key, state)
    return parsed


def inventory_agent(state: OrderState, db_ag: DBAgent):
    key = f"order:{state['order_id']}"
    state = state_store.load_state(key) or state
    print('value fetched from Redis: ', key, state)
    stock = db_ag.get_stock(state["item"])
    prompt = inventory_prompt.format(
        item_in=state["item"],
        qty_in=state["qty"],
        order_id_in=state["order_id"],
        stock_in=stock
                                 )

    logging.debug(f"Inventory Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = llm.invoke(prompt)
    et = time.time()
    print(f"Inventory Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")
    logging.debug(f"Inventory Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")

    print(f"Inventory Agent: LLM Raw Response Content is \n {response} \n ...")
    logging.debug(f"Inventory Agent: LLM Raw Response Content is \n {response} \n ...")

    parsed = parse_json_response(
        response,
        fallback={"order_id": state["order_id"], "status": "out_of_stock"},
    )

    print(f"Inventory Agent: Current Stock: {stock}, Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")
    logging.debug(f"Inventory Agent: Current Stock: {stock}, Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")

    with open(report_file_name, "a") as f:
        f.write(f"\tInventory Agent: {parsed}\n")
    # If reserved, decrement stock
    if parsed["status"] == "reserved":
        db_ag.update_stock(state["item"], state["qty"])

    # inject delay in reservation response
    print('Delay injected, waiting for response of reservation ...')
    time.sleep(delay)

    state['status'] = parsed['status']
    state_store.save_state(key, state)
    return parsed


# 5. Graph definition
def build_graph(db_ag: DBAgent):
    print('............................ Start Running the Graph ..................')
    workflow = StateGraph(OrderState)

    workflow.add_node("order_agent", lambda state: order_agent(state, db_ag))
    workflow.add_node("inventory_agent", lambda state: inventory_agent(state, db_ag))

    workflow.set_entry_point("order_agent")

    # Conditional edges from order_agent
    def route_from_order(state: OrderState):
        if state["status"] == "INIT":
            return "inventory_agent"
        elif state["status"] in ["reserved", "out_of_stock"]:
            return END
        else:
            return END

    workflow.add_conditional_edges(source="order_agent", path=route_from_order)

    # inventory always sends result back to order_agent
    workflow.add_edge(start_key="inventory_agent", end_key="order_agent")

    return workflow.compile()


def run_trial(idx, db_mode, delay=0):
    """Run one LLM-MAS trial and return metrics."""
    # CPU + memory before
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss  # in bytes
    t1 = time.time()

    if db_mode == 'REAL':
        client = MongoClient("mongodb://localhost:27017/")
        db = client["retail_mas"]
        orders = db.orders
        inventory = db.inventory
    else:
        db = None

    db_agent = DBAgent(db=db, mode=db_mode)
    graph = build_graph(db_agent)

    initial_state: OrderState = {"order_id": str(uuid.uuid4()), "item": item, "qty": 2, "status": "INIT"}
    print(f'Trial {idx}, initial_state is {initial_state}')

    state_store.save_state(f"order:{initial_state['order_id']}", initial_state)
    # Run graph
    result = graph.invoke(initial_state)

    # CPU + memory after
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss
    t2 = time.time()
    cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    mem_used = mem_end - mem_start
    elapsed = t2 - t1

    metrics = {
        "trial": idx,
        "delay": delay,
        "response_time": round((elapsed), 3),
        "cpu_time": round(cpu_used, 5),
        "memory_change": round(mem_used/1024,2),
        "final_status": result,
    }
    return metrics


def parallel_trials(n_trials=10, db_mode="REAL", delay=0, report_file_name="mas_parallel_report.txt"):
    """Run N parallel LLM-MAS trials and log metrics."""
    with open(report_file_name, "w") as f:
        f.write("trial,delay,response_time,cpu_time,memory_change,final_status\n")

    results = []
    with ThreadPoolExecutor(max_workers=int(n_trials/10) + delay) as executor:
        futures = {executor.submit(run_trial, i, db_mode, delay): i for i in range(1, n_trials+1)}
        for future in as_completed(futures):
            metrics = future.result()
            results.append(metrics)
            logging.debug(f"Trial {metrics['trial']} finished: {metrics}")
            print(f"Trial {metrics['trial']} result:", metrics)

            with open(report_file_name, "a") as f:
                f.write(f"{metrics['trial']}    |   {metrics['delay']}  |   {metrics['response_time']}  |   "
                        f"{metrics['cpu_time']} |   {metrics['memory_change']}  |   {metrics['final_status']}\n----------------\n\n")

    return results


def sequential_trials(n_trials=10, db_mode="REAL", delay=0, report_file_name="mas_sequential_report.txt"):
    """Run N parallel LLM-MAS trials and log metrics."""
    with open(report_file_name, "w") as f:
        f.write("trial,delay,response_time,cpu_time,memory_change,final_status\n")

    results = []
    for i in range(1, n_trials+1):
        metrics = run_trial(i, db_mode, delay)
        results.append(metrics)
        logging.debug(f"Trial {metrics['trial']} finished: {metrics}")
        print(f"Trial {metrics['trial']} result:", metrics)

        with open(report_file_name, "a") as f:
            f.write(f"{metrics['trial']}    |   {metrics['delay']}  |   {metrics['response_time']}  |   "
                    f"{metrics['cpu_time']} |   {metrics['memory_change']}  |   {metrics['final_status']}\n----------------\n\n")

    return results


if __name__ == "__main__":
    db_mode = 'REAL'  # REAL | MOCK
    delay = 0
    n_trials = 10
    item = "laptop"
    init_stock = 10
    if db_mode == 'REAL':
        reset_db(item, init_stock)
        input('Check DB state is clean, press any key to continue ...')

    report_file_name="mas_parallel_report.txt"
    sequential_trials(n_trials=n_trials, delay=delay, report_file_name=report_file_name)

    # report_file_name="mas_parallel_report.txt"
    # parallel_trials(n_trials=n_trials, delay=delay, report_file_name=report_file_name)

    final_stock = get_final_stock(item=item)
    print(f'final_stock is: {final_stock}')