import uuid
import json
import time
import re
import logging
import os
import random
import psutil
from typing import TypedDict, Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# external deps:
# pip install pymongo redis psutil langchain_community
from pymongo import MongoClient
import redis

# You mentioned using Ollama:
# from langchain_community.llms import Ollama
# We'll import Ollama if available; otherwise the code will still be structured so you can plug in another client.
try:
    from langchain_community.llms import Ollama  # type: ignore
    ollama_available = True
except Exception:
    Ollama = None  # type: ignore
    ollama_available = False

# Logging
logging.basicConfig(
    filename='llm_mas_sc1_traces.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

process = psutil.Process(os.getpid())
LOCK = threading.Lock()

# ---------------------------
# Configurable globals
# ---------------------------
# Default experiment params (can be overridden via env or when calling functions)
REPORT_FILE = os.environ.get("REPORT_FILE", "exp1_results.txt")
DB_MODE = os.environ.get("DB_MODE", "REAL")  # REAL or MOCK
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASS = os.environ.get("REDIS_PASS", "1")
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
db_name = "retail_exp1"

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2")
# Fault injection:
delay = float(os.environ.get("DELAY", "0"))             # seconds to sleep inside inventory agent
drop_rate = int(os.environ.get("DROP_RATE", "0"))       # percent 0-100
# Experiment defaults:
item = os.environ.get("ITEM", "laptop")
init_stock = int(os.environ.get("INIT_STOCK", "10"))
qty = int(os.environ.get("QTY", "2"))
n_trials = 10
max_workers = n_trials / 1
parallel = True
atomic_update = False

# ---------------------------
# Redis state store (simple)
# ---------------------------
class RedisStateStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASS):
        # If your environment needs auth or non-default port, change above
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)

    def save_state(self, key: str, state: dict, ex: int = 300):
        """Persist state as JSON under a given key"""
        try:
            self.r.set(key, json.dumps(state), ex=ex)
        except Exception as e:
            logging.error(f"Redis save_state error: {e}")

    def load_state(self, key: str) -> dict:
        """Fetch state JSON and return as dict"""
        try:
            data = self.r.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            logging.error(f"Redis load_state error: {e}")
            return {}

state_store = RedisStateStore()

# ---------------------------
# DB agent (REAL or MOCK)
# ---------------------------
class DBTool:
    def __init__(self, db=None, mode: str = 'MOCK'):
        self.mode = mode
        if mode == "REAL":
            if db is None:
                raise ValueError("MongoDB client must be provided in REAL mode")
            self.db = db
            self.orders = db.orders
            self.inventory = db.inventory
        else:
            self.orders: List[Dict[str, Any]] = []
            self.inventory: List[Dict[str, Any]] = []

    def save_order(self, order_id: str, item: str, qty: int):
        if self.mode == "REAL":
            self.orders.insert_one({
                "_id": order_id,
                "item": item,
                "qty": qty,
                "status": "INIT",
                "created_at": time.time()
            })
        else:
            self.orders.append({
                "_id": order_id,
                "item": item,
                "qty": qty,
                "status": "INIT",
                "created_at": time.time()
            })

    def update_order(self, order_id: str, status: str):
        if self.mode == "REAL":
            # choose between atomic update or regular update
            if atomic_update:
                with LOCK:
                    self.orders.update_one({"_id": order_id}, {"$set": {"status": status}})
            else:
                self.orders.update_one({"_id": order_id}, {"$set": {"status": status}})
        else:
            for order in self.orders:
                if order["_id"] == order_id:
                    order["status"] = status
                    break

    def get_stock(self, item: str) -> int:
        if self.mode == "REAL":
            if atomic_update:
                with LOCK:
                    stock = self.inventory.find_one({"item": item})
            else:
                stock = self.inventory.find_one({"item": item})
            if not stock:
                stock = {"item": item, "stock": 0}
                self.inventory.insert_one(stock)
            return stock["stock"]
        else:
            stock = next((s for s in self.inventory if s["item"] == item), None)
            if not stock:
                stock = {"item": item, "stock": 0}
                self.inventory.append(stock)
            return stock["stock"]

    def update_stock(self, item: str, qty_delta: int):
        # qty_delta is positive number of qty to decrement (we use -qty in db update)
        if self.mode == "REAL":
            # choose between atomic update or regular update
            if atomic_update:
                with LOCK:
                    self.inventory.find_one_and_update({"item": item, "stock": {"$gte": qty_delta}},
                        {"$inc": {"stock": -qty_delta}})
            else:
                self.inventory.update_one({"item": item}, {"$inc": {"stock": -qty_delta}}, upsert=True)
        else:
            for s in self.inventory:
                if s["item"] == item:
                    s["stock"] -= qty_delta
                    return
            # if not found, create negative stock (shouldn't happen in normal use)
            self.inventory.append({"item": item, "stock": -qty_delta})

# ---------------------------
# Helpers: DB connect/reset/audit
# ---------------------------
def real_db():
    client = MongoClient(MONGO_URL)
    db = client[db_name]
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
    expected_total_reserved = int((init_stock) / qty)  # approximate expectation from your earlier code
    if stock_left < 0:
        failure_rate += -stock_left / qty
        final_ec_state = "FAIL"
    elif stock_left + total_completed_orders != expected_total_reserved:
        failure_rate += abs((total_completed_orders - (expected_total_reserved - stock_left)))
        final_ec_state = "FAIL"
    if total_pending_orders > 0:
        failure_rate += total_pending_orders
        final_ec_state = "FAIL"
    return stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved, final_ec_state, failure_rate

# ---------------------------
# LLM setup & prompts
# ---------------------------
# Use Ollama if available (your original) otherwise a dummy echo LLM for development.
if ollama_available:
    try:
        llm = Ollama(model=OLLAMA_MODEL)
    except Exception as e:
        logging.warning(f"Ollama init failed: {e}")
        llm = None
else:
    llm = None

# Order agent prompt template
order_prompt_template = """
You are the Order Agent.
Decide what to do with an incoming order.

Status: {status_in}

Rules:
- If Status is empty or INIT → return JSON: {{"order_id": "<id>", "status": "INIT", "forward": true}}
- If Status is reserved or out_of_stock or error → return JSON: {{"order_id": "<id>", "status": "<status>", "forward": false}}

Return only one JSON object.
"""

# Inventory agent prompt template
inventory_prompt_template = """
You are the Inventory Agent.
Decide whether to reserve stock.

Order ID: {order_id_in}
Item: {item_in}
Quantity: {qty_in}
Stock: {stock_in}

Rules:
- If Stock >= Quantity → reserved
- Otherwise → out_of_stock

Return JSON with keys: order_id, status (reserved/out_of_stock).
"""

def call_llm(prompt: str) -> str:
    """Call the configured LLM and return a text response. If no LLM available, return a fallback JSON."""
    if llm is None:
        # fallback deterministic behavior: simple rules
        # if prompt contains "Stock:" and a number, decide accordingly
        m = re.search(r"Stock:\s*(\d+)", prompt)
        mqty = re.search(r"Quantity:\s*(\d+)", prompt)
        if m and mqty:
            stock_val = int(m.group(1))
            q = int(mqty.group(1))
            if stock_val >= q:
                return json.dumps({"order_id": str(uuid.uuid4()), "status": "reserved", "forward": False})
            else:
                return json.dumps({"order_id": str(uuid.uuid4()), "status": "out_of_stock", "forward": False})
        # fallback for order agent
        return json.dumps({"order_id": str(uuid.uuid4()), "status": "INIT", "forward": True})
    # If Ollama is available, call it. Use try/except to guard.
    try:
        # Ollama API may differ by version; user's previous code used llm.invoke(prompt)
        resp = llm.invoke(prompt)
        # resp may be a string, or an object; coerce to string
        if isinstance(resp, (dict, list)):
            return json.dumps(resp)
        return str(resp)
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        # fallback deterministic behavior
        return json.dumps({"order_id": str(uuid.uuid4()), "status": "error", "forward": False})

def parse_json_response(content: str, fallback: dict):
    """Extract JSON object from LLM output, safely parse it."""
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # If no JSON, attempt to parse a bare token like "yes" -> map to JSON
        text = content.strip().lower()
        if text in ("yes", "reserved"):
            return {"status": "reserved"}
        if text in ("no", "out_of_stock"):
            return {"status": "out_of_stock"}
        return fallback
    except Exception as e:
        logging.error(f"JSON parse error: {e} -- content: {content}")
        return fallback

# ---------------------------
# Typed state
# ---------------------------
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]  # INIT, reserved, out_of_stock, error
    forward: Optional[bool]

# ---------------------------
# Agent implementations (return PARTIAL dicts)
# ---------------------------

def order_agent(state: OrderState, db_tool: DBTool):
    """
    Order agent: consults LLM to decide action.
    Returns partial dict like: {"order_id": "...", "status":"INIT", "forward": True}
    """
    key = f"order:{state['order_id']}"
    existing = state_store.load_state(key) or state
    # Merge incoming state with persisted state (persisted overrides missing fields)
    merged = {**state, **existing}
    # If status was set in persisted state, use it; else use incoming
    status_in = merged.get("status") or "INIT"

    prompt = order_prompt_template.format(status_in=status_in)
    logging.debug(f"Order Agent PROMPT: {prompt}")

    st = time.time()
    response_text = call_llm(prompt)
    et = time.time()
    logging.debug(f"Order Agent LLM latency: {et - st:.3f}s, raw response: {response_text}")

    parsed = parse_json_response(response_text, fallback={
        "order_id": merged.get("order_id") or str(uuid.uuid4()),
        "status": "INIT",
        "forward": True
    })

    # preserve original order_id if provided
    order_id = merged.get("order_id") or parsed.get("order_id") or str(uuid.uuid4())

    # Persist INIT order when LLM asks for INIT (save as partial DB action, done by DBTool)
    if str(parsed.get("status", "")).lower() == "init":
        # Save order with INIT status if not already present
        db_tool.save_order(order_id, merged.get("item"), merged.get("qty"))

    # Save merged state in Redis
    partial = {
        "order_id": order_id,
        "status": parsed.get("status", status_in),
        "forward": parsed.get("forward", False)
    }
    state_store.save_state(key, {**merged, **partial})
    # Return partial update (Option B)
    return partial

def inventory_agent(state: OrderState, db_tool: DBTool):
    """
    Inventory agent: consults LLM to decide if reservation should succeed.
    Returns partial dict like: {"status": "reserved"} or {"status": "out_of_stock"} or {"status":"error"}.
    Injects delay and drop_rate fault behavior.
    """
    key = f"order:{state['order_id']}"
    existing = state_store.load_state(key) or state
    merged = {**state, **existing}

    stock = db_tool.get_stock(merged["item"])
    prompt = inventory_prompt_template.format(
        order_id_in=merged["order_id"],
        item_in=merged["item"],
        qty_in=merged["qty"],
        stock_in=stock
    )
    logging.debug(f"Inventory Agent PROMPT: {prompt}")

    st = time.time()
    response_text = call_llm(prompt)
    et = time.time()
    logging.debug(f"Inventory Agent LLM latency: {et - st:.3f}s, raw response: {response_text}")

    parsed = parse_json_response(response_text, fallback={"order_id": merged["order_id"], "status": "out_of_stock"})

    # If LLM says reserved, decrement stock (DB-level op)
    # But before committing, we may inject delay/drop
    # Inject artificial delay
    if delay and delay > 0:
        logging.debug(f"Inventory Agent: injecting delay of {delay}s")
        time.sleep(delay)

    # Simulate drop/failure (returning error and not updating DB)
    if drop_rate and drop_rate > 0 and random.randint(0, 99) < drop_rate:
        logging.warning(f"FAULT INJECTED: drop_rate triggered ({drop_rate}%) on order {merged['order_id']}")
        state_store.save_state(key, {**merged, "status": "error", "forward": False})
        return {"status": "error", "forward": False}

    # If LLM recommended reserved, apply DB update
    status_out = parsed.get("status", "out_of_stock")
    if str(status_out).lower() == "reserved":
        # do the DB update
        db_tool.update_stock(merged["item"], merged["qty"])
        db_tool.update_order(merged["order_id"], "reserved")
        partial = {"status": "reserved", "forward": False}
    else:
        db_tool.update_order(merged["order_id"], "out_of_stock")
        partial = {"status": "out_of_stock", "forward": False}

    # Save partial merged state to Redis
    state_store.save_state(key, {**merged, **partial})
    return partial

# ---------------------------
# Simple "StateGraph" like workflow builder (keeping your earlier API)
# ---------------------------
# Your earlier code used a StateGraph from langgraph.graph with methods:
#   add_node(name, func), add_conditional_edges(source='order_agent', path=route), set_entry_point("order_agent"),
#   add_edge(start_key="inventory_agent", end_key="order_agent"), compile().invoke(initial_state)
#
# We'll implement a thin wrapper that mimics those calls but keeps code self-contained
# (so you don't need to import langgraph if you run locally).
#
# The wrapper behaves: nodes are functions that accept (state, db_tool) and return partial dicts.
# The compiled graph.invoke(initial_state) will run sequence until END.
#
END = "END"

class SimpleStateGraph:
    def __init__(self):
        self.nodes = {}
        self.entry = None
        self.conditional_routes = {}
        self.edges = {}  # direct edges mapping start -> end (single follow)
    def add_node(self, name: str, fn):
        self.nodes[name] = fn
    def set_entry_point(self, name: str):
        self.entry = name
    def add_conditional_edges(self, source: str, path):
        # path is a function taking state and returning next node name or END
        self.conditional_routes[source] = path
    def add_edge(self, start_key: str, end_key: str):
        # single directed edge
        self.edges[start_key] = end_key
    def compile(self):
        # return an object with invoke(initial_state) method
        graph = self
        class Compiled:
            def __init__(self, graph):
                self.graph = graph
            def invoke(self, initial_state: Dict[str, Any], db_tool: DBTool):
                # state is a dict and persists across steps via Redis
                current_node = graph.entry
                state = initial_state.copy()
                state_key = f"order:{state['order_id']}"
                state_store.save_state(state_key, state)
                # run until END
                visited = 0
                MAX_VISITS = 100
                last_partial = {}
                while current_node and current_node != END and visited < MAX_VISITS:
                    visited += 1
                    fn = graph.nodes.get(current_node)
                    if fn is None:
                        logging.error(f"Graph invoke: node not found: {current_node}")
                        break
                    # call node function (fn may expect (state, db_tool))
                    partial = fn(state, db_tool)
                    # merge partial into state (Option B)
                    state.update(partial or {})
                    state_store.save_state(state_key, state)
                    # decide next node
                    # if there is a conditional route from current_node
                    if current_node in graph.conditional_routes:
                        route_fn = graph.conditional_routes[current_node]
                        next_node = route_fn(state)
                    elif current_node in graph.edges:
                        next_node = graph.edges[current_node]
                    else:
                        next_node = END
                    # update
                    last_partial = partial
                    current_node = next_node
                return state
        return Compiled(graph)

# ---------------------------
# Build workflow using wrapper
# ---------------------------
def build_graph(db_tool: DBTool):
    workflow = SimpleStateGraph()
    workflow.add_node("order_agent", lambda st, db: order_agent(st, db))
    workflow.add_node("inventory_agent", lambda st, db: inventory_agent(st, db))
    workflow.set_entry_point("order_agent")

    def route_from_order(state: OrderState):
        s = state.get("status", "").upper() if state.get("status") else "INIT"
        # if INIT -> go to inventory_agent (forward True)
        if s == "INIT" or state.get("forward", True):
            return "inventory_agent"
        # if reserved/out_of_stock/error -> END
        if s in ["RESERVED", "OUT_OF_STOCK", "ERROR"]:
            return END
        return END

    workflow.add_conditional_edges(source="order_agent", path=route_from_order)
    # inventory agent always sends result back to order agent (which will read state and decide)
    workflow.add_edge(start_key="inventory_agent", end_key="order_agent")
    return workflow.compile()

# ---------------------------
# Runner: single trial + parallel/sequential harness
# ---------------------------
def run_trial(idx, db_mode='REAL', delay_s=0, drop_pct=0, report_file_name=REPORT_FILE):
    # set globals for this trial
    global delay, drop_rate
    delay = delay_s
    drop_rate = drop_pct

    # resource usage baseline
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss
    t1 = time.time()

    if db_mode == 'REAL':
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
    else:
        db = None

    db_tool = DBTool(db=db, mode=db_mode)

    # If REAL, DB should be pre-reset externally or via reset_db before calling trials
    graph = build_graph(db_tool)

    initial_state: OrderState = {"order_id": str(uuid.uuid4()), "item": item, "qty": qty, "status": "INIT", "forward": True}
    state_store.save_state(f"order:{initial_state['order_id']}", initial_state)

    result_state = graph.invoke(initial_state, db_tool)

    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss
    t2 = time.time()
    cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    mem_used = mem_end - mem_start
    elapsed = t2 - t1

    metrics = {
        "trial": idx,
        "delay": delay,
        "drop_rate": drop_rate,
        "response_time": round(elapsed, 3),
        "cpu_time": round(cpu_used, 5),
        "memory_change_bytes": mem_used,
        "n_threads": max_workers,
        "final_state": result_state
    }

    # Append to report file
    with open(report_file_name, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    return metrics

def sequential_trials(n_trials=n_trials, db_mode="REAL", delay_s=0.0, drop_pct=0, report_file_name=REPORT_FILE):
    with open(report_file_name, "w") as f:
        f.write("")  # reset
    results = []
    for i in range(1, n_trials + 1):
        m = run_trial(i, db_mode=db_mode, delay_s=delay_s, drop_pct=drop_pct, report_file_name=report_file_name)
        results.append(m)
        logging.debug(f"Trial {i} finished: {m}")
        print(f"Trial {i} result: {m}")
    return results

def parallel_trials(n_trials=n_trials, db_mode="REAL", delay_s=0.0, drop_pct=0, max_workers=max_workers,
                    report_file_name=REPORT_FILE):
    with open(report_file_name, "w") as f:
        f.write("")  # reset
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_trial, i, db_mode, delay_s, drop_pct, report_file_name): i for i in range(1, n_trials + 1)}
        for future in as_completed(futures):
            m = future.result()
            results.append(m)
            logging.debug(f"Trial {m['trial']} finished: {m}")
            print(f"Trial {m['trial']} result: {m}")
    return results

# ---------------------------
# Main: run a small experiment if executed directly
# ---------------------------
if __name__ == "__main__":
    # pick experiment params; override via ENV if desired
    DB_MODE = os.environ.get("DB_MODE", DB_MODE)  # REAL or MOCK
    delay = float(os.environ.get("DELAY", delay))
    drop_rate = int(os.environ.get("DROP_RATE", drop_rate))
    n_trials = int(os.environ.get("N_TRIALS", n_trials))
    parallel = os.environ.get("PARALLEL", parallel)

    with open(REPORT_FILE, "w") as f:
        f.write("")

    print(f"Starting LLM-MAS experiment: DB_MODE={DB_MODE}, delay={delay}s, drop_rate={drop_rate}%, n_trials={n_trials}, parallel={parallel}")
    logging.info(f"Experiment start: DB_MODE={DB_MODE}, delay={delay}, drop_rate={drop_rate}, n_trials={n_trials}")

    if DB_MODE == 'REAL':
        reset_db(item, init_stock)
        input("DB reset. Press Enter to start the trials...")

    if parallel:
        results = parallel_trials(n_trials=n_trials, db_mode=DB_MODE, delay_s=delay, drop_pct=drop_rate, max_workers=max_workers)
    else:
        results = sequential_trials(n_trials=n_trials, db_mode=DB_MODE, delay_s=delay, drop_pct=drop_rate)

    # If REAL, print final summary
    if DB_MODE == 'REAL':
        stock_left, total_completed_orders, total_pending_orders, total_oos_orders, expected_total_reserved,\
            final_ec_state, failure_rate = get_final_state(item)
        summary = {
            "n_trials": n_trials,
            "n_threads": max_workers,
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

        with open("exp1_results.json", "w") as f:
            json.dump({"trial_results": results, "final_summary": summary}, f, indent=4)

    print("Done. Results saved to exp1_results.json and detailed lines in", REPORT_FILE)
