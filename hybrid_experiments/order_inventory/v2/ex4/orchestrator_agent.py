import uuid
import time
import json
import re
import logging
import redis
import httpx
import os
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from typing import TypedDict, Optional
import concurrent.futures
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

# -----------------------
# Environment variables
# -----------------------
ORDER_SERVICE_URL = os.getenv("ORDER_SERVICE_URL", "http://localhost:8001")
INVENTORY_SERVICE_URL = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8002")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASS = os.getenv("REDIS_PASS", "1")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "retail_exp4")
ORDERS_COLL = os.environ.get("ORDERS_COLL", "orders")
INVENTORY_COLL = os.environ.get("INVENTORY_COLL", "inventory")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2")
DEFAULT_ITEM = os.getenv("DEFAULT_ITEM", "laptop")
DEFAULT_QTY = int(os.getenv("DEFAULT_QTY", 2))
INIT_STOCK = int(os.environ.get("INIT_STOCK", "10"))

N_TRIALS = int(os.getenv("N_TRIALS", 10))
NUM_WORKERS = N_TRIALS

# -----------------------
# Redis State Store
# -----------------------
class RedisStateStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASS):
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)

    def save_state(self, key: str, state: dict):
        self.r.set(key, json.dumps(state), ex=300)

    def load_state(self, key: str) -> dict:
        data = self.r.get(key)
        return json.loads(data) if data else {}

state_store = RedisStateStore()

# -----------------------
# LLM
# -----------------------
llm = Ollama(model=LLM_MODEL)

def call_llm(prompt: str) -> dict:
    resp = llm.invoke(prompt)
    match = re.search(r"\{.*\}", resp, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}


def real_db():
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    return client, db


def reset_db(item_name: str, init_stock_value: int):
    client, db = real_db()
    orders = db.orders
    inventory = db.inventory
    orders.delete_many({})
    inventory.delete_many({})
    inventory.insert_one({"item": item_name, "stock": init_stock_value})


def get_final_state(item_name: str):
    """
    Returns:
      stock_left, total_completed_orders(reserved), total_pending_orders, final_ec_state ('SUCCESS'/'FAIL'), failure_rate (float)
    """
    client, db = real_db()
    final_stock = db.inventory.find_one({"item": item_name})
    stock_left = final_stock["stock"] if final_stock else 0
    total_completed_orders = db.orders.count_documents({"status": "reserved"})
    total_pending_orders = db.orders.count_documents({"status": "INIT"})
    total_oos_orders = db.orders.count_documents({"status": "out_of_stock"})
    # basic heuristics used previously: compute failure rate loosely
    final_ec_state = "SUCCESS"
    failure_rate = 0.0
    expected_total_reserved = int((INIT_STOCK) / DEFAULT_QTY)  # approximate expectation from your earlier code
    if stock_left < 0:
        failure_rate += -stock_left / DEFAULT_QTY
        final_ec_state = "FAIL"
    elif stock_left + total_completed_orders != expected_total_reserved:
        failure_rate += abs((total_completed_orders - (expected_total_reserved - stock_left)))
        final_ec_state = "FAIL"
    if total_pending_orders > 0:
        failure_rate += total_pending_orders
        final_ec_state = "FAIL"
    return stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved,\
           final_ec_state, failure_rate


# -----------------------
# Tool/Microservice calls
# -----------------------
def tool_create_order(order_id, item, qty):
    with httpx.Client(timeout=5) as c:
        r = c.post(f"{ORDER_SERVICE_URL}/create", json={"order_id": order_id, "item": item, "qty": qty})
        r.raise_for_status()
        return r.json()


def tool_reserve_inventory(order_id, item, qty):
    with httpx.Client(timeout=5) as c:
        r = c.post(f"{INVENTORY_SERVICE_URL}/reserve", json={"order_id": order_id, "item": item, "qty": qty})
        r.raise_for_status()
        return r.json()


def tool_update_order_status(order_id, status):
    with httpx.Client(timeout=5) as c:
        r = c.post(f"{ORDER_SERVICE_URL}/update", json={"order_id": order_id, "status": status})
        r.raise_for_status()
        return r.json()


# -----------------------
# Order State
# -----------------------
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]
    history: list
    trace: list


# -----------------------
# Orchestrator Agent Node
# -----------------------
def orchestrator_node(state: OrderState):
    key = f"orchestrator_node:{state['order_id']}"
    state = state_store.load_state(key) or state
    if "history" not in state: state["history"] = []
    if "trace" not in state: state["trace"] = []

    prompt = f"""You are Orchestrator Agent for two microservices: order and inventory, Based on the current status, choose the correct action: 
             if status is null: choose action as create_order 
             else if status is INIT: choose action as reserve_inventory
             else if status is in (reserved, out_of_stock) : choose action as update_order_status
             else choose action as None
             Finally return a JSON response as: {{"action":"<action>"}}
             Current status: {state['status']}"""
    t1 = time.time()
    print(f' --> Orchestrator Agent reasoning: \n {prompt} \n----------------------------\n')
    llm_resp = call_llm(prompt)
    t2 = time.time()
    print(f' --> Orchestrator Agent reasoning response Took {round((t2-t1),3)}: \n {llm_resp}'
          f' \n----------------------------\n')

    action = llm_resp.get("action")
    print(f"Action chosen by Orchestrator agent: {action}")
    state["trace"].append({"step": "orchestrator_agent_reasoning", "out": llm_resp, "took": round((t2 - t1), 3)})

    if action == "create_order":
        t1 = time.time()
        out = tool_create_order(state["order_id"], state["item"], state["qty"])
        t2 = time.time()
        state["trace"].append({"step": "create_order", "out": out, "took": round((t2 - t1), 3)})
        state["status"] = out.get("status", None)
        state["history"].append({"role": "create_order_tool", "content": json.dumps(out)})
    elif action == "reserve_inventory":
        # Reserve stock in inventory
        t1 = time.time()
        out = tool_reserve_inventory(state["order_id"], state["item"], state["qty"])
        t2 = time.time()
        state["trace"].append({"step": "reserve_inventory", "out": out, "took": round((t2 - t1), 3)})
        state["status"] = out.get("status", "INIT")
        state["history"].append({"role": "reserve_inventory_tool", "content": json.dumps(out)})
    elif action == "update_order_status":
        # Update order status in Order Service
        t1 = time.time()
        status_update = tool_update_order_status(state["order_id"], state["status"])
        t2 = time.time()
        state["status"] = "completed"
        state["trace"].append({"step": "update_order_status", "out": status_update, "took": round((t2 - t1), 3)})
        state["history"].append({"role": "update_order_status_tool", "content": json.dumps(status_update)})

    state_store.save_state(key, state)
    return state


# -----------------------
# Build Graph
# -----------------------
def build_graph():
    workflow = StateGraph(OrderState)
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.set_entry_point("orchestrator_agent")

    # Conditional edges
    def order_next(state: OrderState):
        if state["status"] in ["INIT", "reserved", "out_of_stock"]:
            return "orchestrator_agent"
        return END

    workflow.add_conditional_edges("orchestrator_agent", path=order_next)
    workflow.add_edge("orchestrator_agent", END)
    return workflow.compile()


# -----------------------
# Parallel Trial Runner
# -----------------------
def run_parallel_trials(n_trials=N_TRIALS):
    results = []
    workflow = build_graph()

    def run_trial(trial_id):
        state = OrderState(
            order_id=str(uuid.uuid4()),
            item=DEFAULT_ITEM,
            qty=DEFAULT_QTY,
            status=None,
            history=[],
            trace=[]
        )
        t1 = time.time()
        result = workflow.invoke(state)
        t2 = time.time()
        result["trial"] = trial_id
        result["elapsed"] = round((t2-t1), 3)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_trial, i) for i in range(1, n_trials+1)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    return results


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    reset_db(item_name=DEFAULT_ITEM, init_stock_value=INIT_STOCK)
    input("DB reset. Press Enter to start the trials...")

    results = run_parallel_trials()

    stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved, \
    final_ec_state, failure_rate = get_final_state(DEFAULT_ITEM)

    summary = {
        "n_trials": N_TRIALS,
        "n_threads": NUM_WORKERS,
        "stock_left": stock_left,
        "total_completed_orders": total_completed_orders,
        "total_pending_orders": total_pending_orders,
        "total_oos_orders": total_oos_orders,
        "expected_total_reserved": expected_total_reserved,
        "final_ec_state": final_ec_state,
        "failure_rate": failure_rate
    }
    print("Final summary:", summary)

    with open("exp4_results.json", "w") as f:
        json.dump({"trial_results": results, "final_summary": summary}, f, indent=4)

