import uuid
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional, Dict, List
from pymongo import MongoClient
import json
import time
import re
import logging
import psutil
import os

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


# 2. Deterministic DB Agent
class DBAgent:
    def __init__(self, db, mode: str = 'MOCK'):
        self.mode = mode
        if mode == "REAL":
            if db is None:
                raise ValueError("MongoDB client must be provided in real mode")
            self.db = db
        elif mode == "MOCK":
            self.orders: List[Dict] = []
            self.inventory: List[Dict] = []
        else:
            raise ValueError("mode must be 'real' or 'mock'")

    def save_order(self, order_id: str, item: str, qty: int):
        if self.mode == "real":
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
        if self.mode == "real":
            self.db.orders.update_one({"_id": order_id}, {"$set": {"status": status}})
        else:
            for order in self.orders:
                if order["_id"] == order_id:
                    order["status"] = status
                    break

    def get_stock(self, item: str) -> int:
        if self.mode == "real":
            stock = self.db.inventory.find_one({"item": item})
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
        if self.mode == "real":
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
  "order_id": "d550ad79-4960-4e8f-8ba7-db9a2a95dd0f",
  "item": "laptop",
  "qty": 2,
  "status": "INIT",
  "forward": true
}}

If Status is reserved:
 ```json
{{
  "order_id": "d550ad79-4960-4e8f-8ba7-db9a2a95dd0f",
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
def order_agent(state: OrderState, db_ag: DBAgent):
    # First or second step, LLM decides what to do
    prompt = order_prompt.format(
        item_in=state["item"],
        qty_in=state["qty"],
        order_id_in=state["order_id"] or str(uuid.uuid4()),
        status_in=state["status"] or "INIT"
                                 )
    logging.debug(f"Order Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = llm.invoke(prompt)
    et = time.time()
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


def inventory_agent(state: OrderState, db_ag: DBAgent):
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
    time.sleep(delay)
    return parsed


# 5. Graph definition
def build_graph(db_ag: DBAgent):
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


if __name__ == "__main__":

    report_file_name = 'mas_sc1_parallel.txt'

    cpu_start = process.cpu_times()
    t1 = time.time()

    with open(report_file_name, 'w') as f:
        f.write('')

    db_mode = 'MOCK'  # REAL | MOCK

    if db_mode == 'REAL':
        client = MongoClient("mongodb://user:pass1@localhost:27017/")
        db = client["retail_mas"]
        orders = db.orders
        inventory = db.inventory
    else:
        db = None
    db_agent = DBAgent(db=db, mode=db_mode)
    graph = build_graph(db_agent)
    delay = 0

    initial_state: OrderState = {"order_id": "", "item": "laptop", "qty": 2, "status": "INIT"}

    # OrderAgent (INIT order)
    #    → InventoryAgent (decide reservation)
    #    → OrderAgent (finalize order status)
    #    → END
    result = graph.invoke(initial_state)

    t2 = time.time()
    cpu_end = process.cpu_times()
    cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

    logging.debug(f"Final state: {result}, Total Response Took: {round((t2 - t1), 3)}")
    print("Final state:", result, f' Total Response Took: {round((t2 - t1), 3)}')
    with open(report_file_name, 'a') as f1:
        f1.write(f'Delay : {delay}, '
                 f'Total Response Took: {round((t2 - t1), 3)}, '
                 f'Status: {result}, '
                 f'CPU time: {round(cpu_used, 5)} \n')

