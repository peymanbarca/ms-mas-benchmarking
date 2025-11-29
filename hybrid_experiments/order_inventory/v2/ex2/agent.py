import os
import uuid
import time
import json
import random
import logging
import psutil
from typing import TypedDict, Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import redis
import httpx
from pymongo import MongoClient

# Optional Ollama import (if you use it). Fallback to deterministic.
try:
    from langchain_community.llms import Ollama  # type: ignore
    ollama_available = True
except Exception:
    Ollama = None
    ollama_available = False

logging.basicConfig(level=logging.INFO, filename="exp2_traces.log", format='%(asctime)s %(levelname)s %(message)s')
process = psutil.Process(os.getpid())

# ---------------------------
# Config
# ---------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASS = os.environ.get("REDIS_PASS", "1")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "retail_exp2")
ORDERS_COLL = os.environ.get("ORDERS_COLL", "orders")
INVENTORY_COLL = os.environ.get("INVENTORY_COLL", "inventory")

INVENTORY_SERVICE_URL = os.environ.get("INVENTORY_SERVICE_URL", "http://localhost:8000/reserve")
SERVICE_RESET_URL = os.environ.get("SERVICE_RESET_URL", "http://localhost:8000/reset")

REPORT_FILE = os.environ.get("REPORT_FILE", "exp2_report.txt")
RESULTS_JSON = os.environ.get("RESULTS_JSON", "result/exp2_results.json")

# Experiment defaults (can override via env)
ITEM = os.environ.get("ITEM", "laptop")
INIT_STOCK = int(os.environ.get("INIT_STOCK", "10"))
QTY = int(os.environ.get("QTY", "2"))

# Fault injection for agent-side (in addition to service)
AGENT_DELAY = float(os.environ.get("AGENT_DELAY", "0"))   # seconds inside agent before calling service
AGENT_DROP = int(os.environ.get("AGENT_DROP", "0"))       # simulate agent-level failure (%)

# LLM model (if using Ollama)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2")

n_trials = 10
max_workers = n_trials / 1
parallel = True
atomic_update = False

# ---------------------------
# Redis state store
# ---------------------------
class RedisStateStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASS):
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
    def save_state(self, key: str, state: dict, ex: int = 300):
        try:
            self.r.set(key, json.dumps(state), ex=ex)
        except Exception as e:
            logging.error(f"Redis save error: {e}")
    def load_state(self, key: str) -> dict:
        try:
            v = self.r.get(key)
            return json.loads(v) if v else {}
        except Exception as e:
            logging.error(f"Redis load error: {e}")
            return {}

state_store = RedisStateStore()

# ---------------------------
# DB wrapper/agent
# ---------------------------

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
    expected_total_reserved = int((INIT_STOCK) / QTY)  # approximate expectation from your earlier code
    if stock_left < 0:
        failure_rate += -stock_left / QTY
        final_ec_state = "FAIL"
    elif stock_left + total_completed_orders != expected_total_reserved:
        failure_rate += abs((total_completed_orders - (expected_total_reserved - stock_left)))
        final_ec_state = "FAIL"
    if total_pending_orders > 0:
        failure_rate += total_pending_orders
        final_ec_state = "FAIL"
    return stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved,\
           final_ec_state, failure_rate


class DBTool:
    def __init__(self, mode: str = "REAL"):
        self.mode = mode
        if mode == "REAL":
            client = MongoClient(MONGO_URL)
            db = client[DB_NAME]
            self.orders = db[ORDERS_COLL]
            self.inventory = db[INVENTORY_COLL]
        else:
            self.orders = []
            self.inventory = []

    def save_order(self, order_id: str, item: str, qty: int):
        if self.mode == "REAL":
            self.orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT", "ts": time.time()})
        else:
            self.orders.append({"_id": order_id, "item": item, "qty": qty, "status": "INIT", "ts": time.time()})

    def update_order(self, order_id: str, status: str):
        if self.mode == "REAL":
            self.orders.update_one({"_id": order_id}, {"$set": {"status": status}})
        else:
            for o in self.orders:
                if o["_id"] == order_id:
                    o["status"] = status
                    break

    def get_stock(self, item: str) -> int:
        if self.mode == "REAL":
            stock = self.inventory.find_one({"item": item})
            return stock["stock"] if stock else 0
        else:
            s = next((x for x in self.inventory if x["item"] == item), None)
            return s["stock"] if s else 0

    def set_stock(self, item: str, stock: int):
        if self.mode == "REAL":
            self.inventory.update_one({"item": item}, {"$set": {"stock": stock}}, upsert=True)
        else:
            s = next((x for x in self.inventory if x["item"] == item), None)
            if s:
                s["stock"] = stock
            else:
                self.inventory.append({"item": item, "stock": stock})

# ---------------------------
# LLM & prompt helpers (fallback deterministic)
# ---------------------------
if ollama_available:
    try:
        llm = Ollama(model=OLLAMA_MODEL)
    except Exception as e:
        logging.warning(f"Ollama init failed: {e}")
        llm = None
else:
    llm = None

order_prompt_template = """
You are the Order Agent.
Given Status: {status_in}
If status is INIT, return JSON: {{"order_id":"<id>","status":"INIT","forward": true}}
If status already final, return JSON with the same status and forward=false.
"""

def call_llm(prompt: str) -> str:
    """Call LLM or fallback deterministic output."""
    if llm is None:
        # deterministic fallback: always INIT
        return json.dumps({"order_id": str(uuid.uuid4()), "status": "INIT", "forward": True})
    try:
        resp = llm.invoke(prompt)
        return str(resp)
    except Exception as e:
        logging.error(f"LLM call error: {e}")
        return json.dumps({"order_id": str(uuid.uuid4()), "status": "error", "forward": False})

def parse_json_response(text: str, fallback: dict):
    import re
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        t = text.strip().lower()
        if t in ("reserved", "yes"):
            return {"status": "reserved"}
        if t in ("out_of_stock", "no"):
            return {"status": "out_of_stock"}
        return fallback
    except Exception as e:
        logging.error(f"parse error: {e} -- {text}")
        return fallback




# ---------------------------
# Typed state
# ---------------------------
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]
    forward: Optional[bool]

# ---------------------------
# Agent: order_agent (LLM) that calls microservice /reserve
# Returns PARTIAL dict (Option B)
# ---------------------------
import httpx

def order_agent(state: OrderState, db_tool: DBTool):
    key = f"order:{state['order_id']}"
    existing = state_store.load_state(key) or state
    merged = {**state, **existing}
    status_in = merged.get("status") or "INIT"

    prompt = order_prompt_template.format(status_in=status_in)
    logging.debug(f"Order Agent prompt: {prompt}")

    t0 = time.time()
    resp_text = call_llm(prompt)
    t1 = time.time()
    logging.debug(f"Order Agent LLM time: {t1 - t0:.3f}s")

    parsed = parse_json_response(resp_text, fallback={"order_id": merged.get("order_id") or str(uuid.uuid4()), "status": "INIT", "forward": True})
    order_id = merged.get("order_id") or parsed.get("order_id") or str(uuid.uuid4())

    # Save INIT order in DB if requested
    if str(parsed.get("status", "")).lower() == "init":
        db_tool.save_order(order_id, merged["item"], merged["qty"])

    # Simulate agent-side delay or drop before calling service
    if AGENT_DELAY and AGENT_DELAY > 0:
        time.sleep(AGENT_DELAY)
    if AGENT_DROP and AGENT_DROP > 0 and random.randint(0,99) < AGENT_DROP:
        logging.warning(f"Agent-level drop triggered for order {order_id}")
        state_store.save_state(key, {**merged, "order_id": order_id, "status": "error", "forward": False})
        return {"order_id": order_id, "status": "error", "forward": False}

    # Call inventory microservice to reserve
    # If microservice fails (HTTP error), treat as error
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(INVENTORY_SERVICE_URL, json={"item": merged["item"], "qty": merged["qty"], "request_id": order_id})
            if r.status_code >= 500:
                logging.error(f"Inventory service error: {r.status_code} {r.text}")
                state_store.save_state(key, {**merged, "order_id": order_id, "status": "error", "forward": False})
                return {"order_id": order_id, "status": "error", "forward": False}
            jr = r.json()
    except Exception as e:
        logging.error(f"Call to inventory service failed: {e}")
        state_store.save_state(key, {**merged, "order_id": order_id, "status": "error", "forward": False})
        return {"order_id": order_id, "status": "error", "forward": False}

    # interpret microservice response
    if jr.get("reserved"):
        # successful reservation: update order status to reserved
        db_tool.update_order(order_id, "reserved")
        partial = {"order_id": order_id, "status": "reserved", "forward": False, "remaining": jr.get("remaining")}
    else:
        db_tool.update_order(order_id, "out_of_stock")
        partial = {"order_id": order_id, "status": "out_of_stock", "forward": False, "remaining": jr.get("remaining")}

    state_store.save_state(key, {**merged, **partial})
    return partial

# ---------------------------
# Simple graph wrapper (same as earlier SimpleStateGraph)
# ---------------------------
END = "END"
class SimpleStateGraph:
    def __init__(self):
        self.nodes = {}
        self.entry = None
        self.conditional_routes = {}
        self.edges = {}
    def add_node(self, name, fn):
        self.nodes[name] = fn
    def set_entry_point(self, name):
        self.entry = name
    def add_conditional_edges(self, source, path_fn):
        self.conditional_routes[source] = path_fn
    def add_edge(self, start_key, end_key):
        self.edges[start_key] = end_key
    def compile(self):
        graph = self
        class Compiled:
            def __init__(self, graph):
                self.graph = graph
            def invoke(self, initial_state, db_tool):
                state = initial_state.copy()
                key = f"order:{state['order_id']}"
                state_store.save_state(key, state)
                current = graph.entry
                visited = 0
                MAX=100
                while current and current != END and visited < MAX:
                    visited += 1
                    fn = graph.nodes.get(current)
                    if not fn:
                        break
                    partial = fn(state, db_tool)
                    if partial:
                        state.update(partial)
                    state_store.save_state(key, state)
                    # route
                    if current in graph.conditional_routes:
                        next_node = graph.conditional_routes[current](state)
                    elif current in graph.edges:
                        next_node = graph.edges[current]
                    else:
                        next_node = END
                    current = next_node
                return state
        return Compiled(graph)

# ---------------------------
# Build workflow
# ---------------------------
def build_graph(db_tool: DBTool):
    g = SimpleStateGraph()
    g.add_node("order_agent", lambda s, db: order_agent(s, db))
    g.set_entry_point("order_agent")
    # routing: if INIT -> call agent which will call service and return final status; graph ends.
    def route_from_order(state):
        s = (state.get("status") or "INIT").upper()
        if s in ["INIT", "ERROR"]:
            # we still invoke node; by design order_agent will call service internally
            return END
        return END
    g.add_conditional_edges("order_agent", route_from_order)
    return g.compile()

# ---------------------------
# Runner and harness
# ---------------------------
def run_trial(idx, db_mode="REAL", report_file=REPORT_FILE, agent_delay=0, agent_drop=0):
    global AGENT_DELAY, AGENT_DROP
    AGENT_DELAY = agent_delay
    AGENT_DROP = agent_drop

    # resources baseline
    cpu0 = process.cpu_times()
    mem0 = process.memory_info().rss
    t0 = time.time()

    db_tool = DBTool(mode=db_mode)
    graph = build_graph(db_tool)

    init_state = {"order_id": str(uuid.uuid4()), "item": ITEM, "qty": QTY, "status": "INIT", "forward": True}
    state_store.save_state(f"order:{init_state['order_id']}", init_state)

    final_state = graph.invoke(init_state, db_tool)

    cpu1 = process.cpu_times()
    mem1 = process.memory_info().rss
    t1 = time.time()
    cpu_used = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    mem_used = mem1 - mem0
    elapsed = t1 - t0

    metrics = {
        "trial": idx,
        "agent_delay": agent_delay,
        "agent_drop": agent_drop,
        "elapsed": round(elapsed, 4),
        "cpu_time": round(cpu_used, 4),
        "mem_bytes": mem_used,
        "n_threads": max_workers,
        "final_state": final_state
    }
    with open(report_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    return metrics

def sequential_trials(n=n_trials, db_mode="REAL", agent_delay=0.0, agent_drop=0, reset_db_before=True):
    if reset_db_before and db_mode == "REAL":
        # call service reset to set initial stock
        try:
            with httpx.Client(timeout=5.0) as c:
                c.post(SERVICE_RESET_URL, json={"item": ITEM, "qty": INIT_STOCK})
        except Exception as e:
            logging.warning(f"Service reset failed: {e}")
    results=[]
    for i in range(1, n+1):
        m = run_trial(i, db_mode=db_mode, agent_delay=agent_delay, agent_drop=agent_drop)
        logging.info(f"Trial {i}: {m}")
        print("Trial", i, "->", m)
        results.append(m)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    return results

def parallel_trials(n=n_trials, max_workers=max_workers, db_mode="REAL", agent_delay=0.0, agent_drop=0, reset_db_before=True):
    if reset_db_before and db_mode == "REAL":
        try:
            with httpx.Client(timeout=5.0) as c:
                c.post(SERVICE_RESET_URL, json={"item": ITEM, "qty": INIT_STOCK})
        except Exception as e:
            logging.warning(f"Service reset failed: {e}")
    results=[]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_trial, i, db_mode, REPORT_FILE, agent_delay, agent_drop): i for i in range(1, n+1)}
        for fut in as_completed(futures):
            m = fut.result()
            logging.info(f"Trial done: {m}")
            print("Trial result:", m)
            results.append(m)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    return results

# ---------------------------
# Auditor helper: compute consistency_error
# ---------------------------
def compute_consistency_error_from_db(item_name=ITEM, initial_stock=INIT_STOCK):
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    inv = db[INVENTORY_COLL].find_one({"item": item_name})
    current_stock = inv["stock"] if inv else 0
    committed = db[ORDERS_COLL].count_documents({"status": "reserved"})
    expected = initial_stock - committed * QTY
    error = abs(expected - current_stock)
    return {"current_stock": current_stock, "committed": committed, "expected": expected, "error": error}

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # Pull config from env if present
    DB_MODE = os.environ.get("DB_MODE", "REAL")
    AGENT_DELAY = float(os.environ.get("AGENT_DELAY", AGENT_DELAY))
    AGENT_DROP = int(os.environ.get("AGENT_DROP", AGENT_DROP))
    n_trials = int(os.environ.get("N_TRIALS", n_trials))
    parallel = os.environ.get("PARALLEL", parallel)

    with open(REPORT_FILE, "w") as f:
        f.write("")

    print("Experiment 2: Agent vs Microservice")
    print(f"DB_MODE={DB_MODE}, SERVICE={INVENTORY_SERVICE_URL}, ITEM={ITEM}, INIT_STOCK={INIT_STOCK}, QTY={QTY},"
          f" n_trials={n_trials}, parallel={parallel}")
    logging.info("Starting Experiment 2")

    # reset service state before runs
    if DB_MODE == 'REAL':
        reset_db(item_name=ITEM, init_stock_value=INIT_STOCK)
        input("DB reset. Press Enter to start the trials...")

    if parallel:
        results = parallel_trials(n=n_trials, max_workers=max_workers, db_mode=DB_MODE, agent_delay=AGENT_DELAY, agent_drop=AGENT_DROP)
    else:
        results = sequential_trials(n=n_trials, db_mode=DB_MODE, agent_delay=AGENT_DELAY, agent_drop=AGENT_DROP)

    # final audit (if REAL)
    if DB_MODE == "REAL":
        audit = compute_consistency_error_from_db()
        print("Final audit:", audit)
        with open(REPORT_FILE, "a") as f:
            f.write("\nFINAL_AUDIT:\n")
            f.write(json.dumps(audit) + "\n")
        stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved,\
            final_ec_state, failure_rate = get_final_state(ITEM)

        summary = {
            "stock_left": stock_left,
            "total_completed_orders": total_completed_orders,
            "total_pending_orders": total_pending_orders,
            "total_oos_orders": total_oos_orders,
            "expected_total_reserved": expected_total_reserved,
            "final_ec_state": final_ec_state,
            "failure_rate": failure_rate
        }
        print("Final summary:", summary)
        with open(REPORT_FILE, "a") as f:
            f.write("\nFINAL_SUMMARY:\n")
            f.write(json.dumps(summary) + "\n")

    print("Done. Results in", RESULTS_JSON, "and", REPORT_FILE)
