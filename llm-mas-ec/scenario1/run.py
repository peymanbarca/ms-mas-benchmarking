import uuid
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from pymongo import MongoClient
import json
import time
import re

# MongoDB setup
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail_mas"]
orders = db.orders
inventory = db.inventory


# 1. Define global state
# State just flows down the chain in a single process
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]  # INIT, reserved, out_of_stock


# 2. Deterministic DB Agent
class DBAgent:
    def __init__(self, db):
        self.db = db

    def save_order(self, order_id, item, qty):
        self.db.orders.insert_one({
            "_id": order_id,
            "item": item,
            "qty": qty,
            "status": "INIT"
        })

    def update_order(self, order_id, status):
        self.db.orders.update_one(
            {"_id": order_id},
            {"$set": {"status": status}}
        )

    def get_stock(self, item):
        stock = self.db.inventory.find_one({"item": item})
        if not stock:
            stock = {"item": item, "stock": 10}
            self.db.inventory.insert_one(stock)
        return stock["stock"]

    def update_stock(self, item, qty):
        self.db.inventory.update_one(
            {"item": item},
            {"$inc": {"stock": -qty}},
            upsert=True
        )

# 3. LLM setup
llm = ChatOpenAI(model="gpt-4o-mini")

order_prompt = ChatPromptTemplate.from_template("""
You are the Order Agent.
Decide what to do with an incoming order.

Item: {item}
Quantity: {qty}
Order ID: {order_id}
Current Status: {status}

Rules:
- If status is empty or INIT → create the order and forward to inventory.
- If status is reserved or out_of_stock → finalize order with given status and update DB.

Return JSON with keys: order_id, item, qty, status, forward (true/false).
""")

inventory_prompt = ChatPromptTemplate.from_template("""
You are the Inventory Agent.
Decide whether to reserve stock.

Order ID: {order_id}
Item: {item}
Quantity: {qty}
Stock: {stock}

Rules:
- If stock >= qty → reserved
- Otherwise → out_of_stock

Return JSON with keys: order_id, status.
""")


def parse_json_response(content: str, fallback: dict):
    """Extract JSON object from LLM output, safely parse it."""
    try:
        # Remove ```json ... ``` or ``` fences if present
        cleaned = re.sub(r"^```.*\n", "", content.strip())   # drop opening ```
        cleaned = re.sub(r"\n```$", "", cleaned)             # drop closing ```
        return json.loads(cleaned)
    except Exception as e:
        print(e)
        return fallback


# 4. Agent functions (sync now)
def order_agent(state: OrderState, db: DBAgent):
    # First or second step, LLM decides what to do
    messages = order_prompt.format_messages(
        item=state["item"],
        qty=state["qty"],
        order_id=state["order_id"] or str(uuid.uuid4()),
        status=state["status"] or "INIT",
    )
    response = llm.invoke(messages)
    parsed = parse_json_response(
        response.content,
        fallback={
            "order_id": state["order_id"] or str(uuid.uuid4()),
            "item": state["item"],
            "qty": state["qty"],
            "status": "INIT",
            "forward": True,
        },
    )

    print(f"Order Agent: Requested Qty: {state['qty']}, LLM Response: {parsed}")

    if parsed["status"] == "INIT":
        # Save INIT order
        db.save_order(parsed["order_id"], parsed["item"], parsed["qty"])

    elif parsed["status"] in ["reserved", "out_of_stock"]:
        # Finalize order based on reservation response
        db.update_order(parsed["order_id"], parsed["status"])

    return parsed


def inventory_agent(state: OrderState, db: DBAgent):
    stock = db.get_stock(state["item"])
    messages = inventory_prompt.format_messages(
        order_id=state["order_id"],
        item=state["item"],
        qty=state["qty"],
        stock=stock,
    )
    response = llm.invoke(messages)
    parsed = parse_json_response(
        response.content,
        fallback={"order_id": state["order_id"], "status": "out_of_stock"},
    )

    print(f"Inventory Agent: Current Stock: {stock}, Qty: {state['qty']}, LLM Response: {parsed}")

    # If reserved, decrement stock
    if parsed["status"] == "reserved":
        db.update_stock(state["item"], state["qty"])

    # inject delay in reservation response
    print('Delay injected, waiting for response of reservation ...')
    time.sleep(3)
    return parsed


# 5. Graph definition
def build_graph(db: DBAgent):
    workflow = StateGraph(OrderState)

    workflow.add_node("order_agent", lambda state: order_agent(state, db))
    workflow.add_node("inventory_agent", lambda state: inventory_agent(state, db))

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

    db = DBAgent(db)
    graph = build_graph(db)

    initial_state: OrderState = {"order_id": "", "item": "laptop", "qty": 2, "status": None}

    # OrderAgent (INIT order)
    #    → InventoryAgent (decide reservation)
    #    → OrderAgent (finalize order status)
    #    → END
    result = graph.invoke(initial_state)
    print("Final state:", result)
