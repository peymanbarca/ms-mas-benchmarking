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
import asyncio


logging.basicConfig(
    filename='llm_mas_sc1_traces.log',  # Specify the log file name
    level=logging.DEBUG,          # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
)

process = psutil.Process(os.getpid())


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
llm = Ollama(model="llama3")

order_prompt = """
You are the Order Agent.
Decide what to do with an incoming order.

Item: {item_in}
Quantity: {qty_in}
Order ID: {order_id_in}
Status: {status_in}

Rules:
- If Status is empty or INIT → create the order in DB and forward to inventory agent.
- If Status is reserved or out_of_stock → finalize order with given status and update DB.
Return only one JSON with keys: order_id, item, qty, status, forward (true/false).

Example responses are:

If Status is empty or INIT:
 ```json
{{
  "order_id": "{order_id_in}",
  "item": "laptop",
  "qty": 2,
  "status": "INIT",
  "forward": true
}}

If Status is reserved:
 ```json
{{
  "order_id": "{order_id_in}",
  "item": "laptop",
  "qty": 2,
  "status": "reserved",
  "forward": false
}}

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
async def order_agent(state: OrderState, db_ag: DBAgent):
    # First or second step, LLM decides what to do
    prompt = order_prompt.format(
        item_in=state["item"],
        qty_in=state["qty"],
        order_id_in=state["order_id"],
        status_in=state["status"] or "INIT"
                                 )
    logging.debug(f"Order Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = await llm.ainvoke(prompt)
    et = time.time()
    print(f"Order Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")
    logging.debug(f"Order Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")

    print(f"Order Agent: LLM Raw Response Content is \n {response} \n ...")
    logging.debug(f"Order Agent: LLM Raw Response Content is \n {response} \n ...")

    parsed = parse_json_response(
        response,
        fallback={
            "order_id": state["order_id"] or str(uuid.uuid4()),
            "item": state["item"],
            "qty": state["qty"],
            "status": "INIT",
            "forward": True,
        },
    )

    print(f"Order Agent: Requested Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")
    logging.debug(f"Order Agent: Requested Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")

    if parsed["status"] == "INIT":
        # Save INIT order
        db_ag.save_order(parsed["order_id"], parsed["item"], parsed["qty"])

    elif parsed["status"] in ["reserved", "out_of_stock"]:
        # Finalize order based on reservation response
        db_ag.update_order(parsed["order_id"], parsed["status"])

    return parsed


async def inventory_agent(state: OrderState, db_ag: DBAgent):
    stock = db_ag.get_stock(state["item"])
    prompt = inventory_prompt.format(
        item_in=state["item"],
        qty_in=state["qty"],
        order_id_in=state["order_id"],
        stock_in=stock
                                 )

    logging.debug(f"Inventory Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = await llm.ainvoke(prompt)
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

    # If reserved, decrement stock
    if parsed["status"] == "reserved":
        db_ag.update_stock(state["item"], state["qty"])

    # inject delay in reservation response
    print('Delay injected, waiting for response of reservation ...')
    await asyncio.sleep(delay)

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


async def run_trial(idx, db_mode, delay=0):
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
    # Run graph
    result = await graph.invoke(initial_state)

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


async def parallel_trials(n_trials=10, db_mode="REAL", delay=0, report_file_name="mas_parallel_report.txt"):
    """Run N parallel LLM-MAS trials and log metrics asynchronously."""
    with open(report_file_name, "w") as f:
        f.write("trial,delay,response_time,cpu_time,memory_change,final_status\n")

    tasks = [asyncio.create_task(run_trial(i, db_mode, delay)) for i in range(1, n_trials+1)]
    results = await asyncio.gather(*tasks)

    for metrics in results:
        logging.debug(f"Trial {metrics['trial']} finished: {metrics}")
        print(f"Trial {metrics['trial']} result:", metrics)

        with open(report_file_name, "a") as f:
            f.write(f"{metrics['trial']}    |   {metrics['delay']}  |   {metrics['response_time']}  |   "
                    f"{metrics['cpu_time']} |   {metrics['memory_change']}  |   {metrics['final_status']}\n")

    return results

    # results = []
    # with ThreadPoolExecutor(max_workers=n_trials) as executor:
    #     futures = {executor.submit(run_trial, i, db_mode, delay): i for i in range(1, n_trials+1)}
    #     for future in as_completed(futures):
    #         metrics = future.result()
    #         results.append(metrics)
    #         logging.debug(f"Trial {metrics['trial']} finished: {metrics}")
    #         print(f"Trial {metrics['trial']} result:", metrics)
    #
    #         with open(report_file_name, "a") as f:
    #             f.write(f"{metrics['trial']}    |   {metrics['delay']}  |   {metrics['response_time']}  |   "
    #                     f"{metrics['cpu_time']} |   {metrics['memory_change']}  |   {metrics['final_status']}\n")
    #
    # return results


async def sequential_trials(n_trials=10, db_mode="REAL", delay=0, report_file_name="mas_sequential_report.txt"):
    """Run N sequential LLM-MAS trials and log metrics."""
    with open(report_file_name, "w") as f:
        f.write("trial,delay,response_time,cpu_time,memory_change,final_status\n")

    results = []
    for i in range(1, n_trials+1):
        metrics = await run_trial(i, db_mode, delay)   # await each trial
        results.append(metrics)
        logging.debug(f"Trial {metrics['trial']} finished: {metrics}")
        print(f"Trial {metrics['trial']} result:", metrics)

        with open(report_file_name, "a") as f:
            f.write(f"{metrics['trial']}    |   {metrics['delay']}  |   {metrics['response_time']}  |   "
                    f"{metrics['cpu_time']} |   {metrics['memory_change']}  |   {metrics['final_status']}\n")

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

    results = asyncio.run(parallel_trials(n_trials=10, db_mode="REAL", delay=0))
    # results = asyncio.run(sequential_trials(n_trials=10, db_mode="REAL", delay=0))

    final_stock = get_final_stock(item=item)
    print(f'final_stock is: {final_stock}')


