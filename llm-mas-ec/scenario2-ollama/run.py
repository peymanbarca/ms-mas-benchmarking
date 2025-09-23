import asyncio
import json
import logging
import time
import uuid
import re
from typing import Optional, TypedDict
# from langchain_openai import ChatOpenAI
import aio_pika
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langgraph.checkpoint.memory import MemorySaver

from pymongo import MongoClient
from typing import TypedDict, Optional, Dict, List, Any
import threading

# ---- Config ----
RABBITMQ_URL = "amqp://guest:guest@localhost/"
REQ_QUEUE = "inventory_requests"
RES_QUEUE = "inventory_responses"

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("mas-async-call")


# ---- State type ----
class OrderState(TypedDict):
    order_id: str
    item: str
    qty: int
    status: Optional[str]  # INIT, WAITING, reserved, out_of_stock
    inventory_response: Optional[dict]


# Create in-memory checkpointer
checkpointer = MemorySaver()


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


def get_all_orders(item):
    client, db = real_db()
    orders = db.orders
    all_orders = list(orders.find({"item": item}))
    return all_orders


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
            raise ValueError("mode must be 'REAL' or 'MOCK'")

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
llm = Ollama(model="qwen2")
# llm = ChatOpenAI(model="gpt-4o-mini")

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


# ---- 1) LangGraph nodes (synchronous functions that publish/return) ----

# We'll create the graph and compile to "app" below; we need access to `app` in the listener to call app.update_state()
app = None  # will be set after graph compiled


async def publish_request(message: dict):
    """Publish JSON message to inventory_requests queue (async)."""
    conn = await aio_pika.connect_robust(RABBITMQ_URL)
    async with conn:
        ch = await conn.channel()
        await ch.default_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key=REQ_QUEUE,
        )


loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()


def submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, loop)


def order_agent(state: OrderState, db_ag: DBAgent):
    """
    Order agent behavior (sync): decide via LLM, persist INIT, then publish request to inventory and set WAITING.
    When the inventory response arrives the LangGraph will be resumed (app.update_state).
    """
    LOG.info("[OrderAgent] state in: %s", state)

    # First or second step, LLM decides what to do
    prompt = order_prompt.replace('status_in', state["status"] or "INIT")
    logging.debug(f"Order Agent: Sending prompt for LLM \n {prompt} \n ...")

    st = time.time()
    response = llm.invoke(prompt)
    #response = response.content
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

    # If this is the initial leg, persist INIT and send request to inventory
    if str(parsed.get("status")).lower() == "init" and parsed.get("forward", True):
        order_id = state["order_id"]
        db_ag.save_order(order_id, state["item"], state["qty"])
        req_msg = {"order_id": order_id, "item": state["item"], "qty": state["qty"]}

        # publish asynchronously (fire-and-forget)
        submit_async(publish_request(req_msg))

        # set waiting status
        parsed["status"] = "WAITING"
        parsed["inventory_response"] = None
        LOG.info("[OrderAgent] published request and returning WAITING for order %s", order_id)
        return parsed

    # Otherwise (this run is after inventory response), finalize order
    if str(parsed.get("status")).lower() in ["reserved", "out_of_stock"] or state.get("inventory_response"):
        # if inventory_response present, use that to finalize the order in DB
        inv = state.get("inventory_response") or {}
        if inv:
            final_status = str(inv.get("status")).lower()
        else:
            final_status = str(parsed.get("status")).lower()
        # map inventory status to order final statuses
        if final_status == "reserved":
            db_ag.update_order(state["order_id"], "RESERVED")
            # mock payment success
            db_ag.update_order(state["order_id"], "COMPLETED")
        elif final_status == "out_of_stock":
            db_ag.update_order(state["order_id"], "FAILED_OUT_OF_STOCK")
        else:
            db_ag.update_order(state["order_id"], "FAILED")
        parsed["status"] = final_status
        LOG.info("[OrderAgent] finalized order %s -> %s", state["order_id"], final_status)
        return parsed

    # default - return state unchanged
    return parsed


def inventory_agent(state: OrderState, db_ag: DBAgent):
    """
    Inventory agent: consume the request (this function is used if called synchronously).
    In our async architecture, inventory will be a separate worker process that consumes requests and publishes responses.
    This function kept for completeness/optional local execution.
    """

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
    #response = response.content
    et = time.time()
    print(f"Inventory Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")
    logging.debug(f"Inventory Agent: LLM Raw Response Took {round((et - st), 3)} \n ...")

    print(f"Inventory Agent: LLM Raw Response Content is \n {response} \n ...")
    logging.debug(f"Inventory Agent: LLM Raw Response Content is \n {response} \n ...")

    parsed = parse_json_response(
        response,
        fallback={"order_id": state["order_id"], "status": "out_of_stock"},
    )

    print(
        f"Inventory Agent: Current Stock: {stock}, Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")
    logging.debug(
        f"Inventory Agent: Current Stock: {stock}, Qty: {state['qty']}, LLM Parsed Response: {parsed}, \n\n----------")

    # If reserved, decrement
    if str(parsed.get("status")).lower() == "reserved":
        db_ag.update_stock(state["item"], state["qty"])

    return parsed


# ---- 2) Build LangGraph, compile, and expose `app` (so listener can update state) ----
def build_graph(db_ag: DBAgent):
    global app
    workflow = StateGraph(OrderState)
    workflow.add_node("order_agent", lambda st: order_agent(st, db_ag))
    workflow.add_node("inventory_agent", lambda st: inventory_agent(st, db_ag))

    workflow.set_entry_point("order_agent")

    # routing: after order_agent, go to inventory_agent if status == WAITING? actually we publish and wait,
    # so we always send to inventory_agent (inventory worker handles request from queue).
    def route_from_order(state: OrderState):
        # If we just set WAITING, the graph can go to END (we will resume later).
        # If we are finalizing (inventory_response present), go to END after order_agent finishes
        if str(state.get("status")).lower() in ["waiting", "reserved", "out_of_stock"]:
            # stop graph here (END). We'll resume when response arrives by calling app.update_state(...)
            return END
        # Otherwise, if status indicates to continue, go to inventory_agent (not strictly used)
        if str(state.get("status")).lower() == "init":
            return "inventory_agent"
        # default end
        return END

    workflow.add_conditional_edges(source="order_agent", path=route_from_order)
    # inventory -> order when inventory called synchronously (not used in async worker case)
    workflow.add_edge("inventory_agent", "order_agent")

    app = workflow.compile(checkpointer=checkpointer)
    return app


# ---- 3) InventoryWorker: consumes reservation requests from REQ_QUEUE and publishes responses in RES_QUEUE ----
async def inventory_worker_task(db_ag: DBAgent, delay_seconds: float = 0.0, drop_rate: float = 0.0):
    conn = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await conn.channel()
    queue = await channel.declare_queue(REQ_QUEUE, durable=True)
    exchange = channel.default_exchange

    LOG.info("[InventoryWorker] Listening for requests on %s", REQ_QUEUE)

    async def on_message(msg: aio_pika.IncomingMessage):
        async with msg.process():
            payload = json.loads(msg.body.decode())
            LOG.info("[InventoryWorker] Received: %s", payload)
            order_id = payload["order_id"]
            item = payload["item"]
            qty = payload["qty"]

            # simulate processing delay
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            # simulate drop
            import random
            if random.random() < drop_rate:
                LOG.info("[InventoryWorker] DROPPING response for %s (simulated)", order_id)
                # do not publish response (simulated drop)
                return

            # decide reservation using inventory_agent logic
            # Here we execute inventory_agent synchronously using db_ag or use dummy LLM
            state = OrderState(order_id=order_id, item=item, qty=qty, status=None, inventory_response=None)
            parsed = inventory_agent(state, db_ag)  # synchronous; could be replaced with llm.invoke

            # if reserved, inventory_agent already updated DB via db_ag.update_stock
            response = {"order_id": order_id, "status": parsed.get("status")}
            await exchange.publish(
                aio_pika.Message(body=json.dumps({'state': state, 'response': response}).encode()),
                routing_key=RES_QUEUE
            )
            LOG.info("[InventoryWorker] Published response: %s", response)

    await queue.consume(callback=on_message)


# ---- 4) LangGraph response listener: consumes inventory_responses and resumes the graph ----
async def langgraph_response_listener_task():
    # This listener will push responses into the running graph by calling app.update_state()
    conn = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await conn.channel()
    queue = await channel.declare_queue(RES_QUEUE, durable=True)

    LOG.info("[LangGraphResponseListener] Listening for inventory responses on %s", RES_QUEUE)

    async def on_message(msg: aio_pika.IncomingMessage):
        async with msg.process():
            payload = json.loads(msg.body.decode())
            LOG.info("[LangGraphResponseListener] Received: %s", payload)
            inventory_response = payload['response']
            # The graph expects state object; we create updated state that includes inventory_response
            updated_state = {
                "order_id": inventory_response["order_id"],
                "item": None,  # optional; graph/order_agent uses saved DB if needed
                "qty": None,
                "status": inventory_response['status'],
                "inventory_response": inventory_response
            }
            # Update the running LangGraph state; this resumes processing for that order
            # app.update_state accepts a partial state dict and will continue the workflow
            try:
                app.update_state({"configurable": {"thread_id": inventory_response["order_id"]}},
                                 {"status": inventory_response["status"]}
                                 )
                LOG.info(
                    f"[LangGraphResponseListener] app.update_state called for {inventory_response['order_id']} : {updated_state}")

                LOG.info('Graph resumed and going to run again ...')
                app.invoke(updated_state, {"thread_id": inventory_response["order_id"]})

            except Exception as e:
                LOG.exception("Failed to update graph state: %s", e)

    await queue.consume(callback=on_message)


# ---- 5) Runner to create graph, start tasks, and submit orders (trials) ----
async def main_runner(n_trials: int = 5, db_mode: str = "REAL",
                      delay_seconds: float = 0.0, drop_rate: float = 0.0):
    # Prepare DB agent
    if db_mode == "REAL":
        client = MongoClient("mongodb://localhost:27017/")
        db = client["retail_mas"]
        db_ag = DBAgent(db=db, mode="REAL")
    else:
        db_ag = DBAgent(db=None, mode="MOCK")

    # Build graph (app)
    build_graph(db_ag)

    # Start inventory worker and response listener
    # They will run forever - run them as background tasks
    asyncio.create_task(inventory_worker_task(db_ag, delay_seconds=delay_seconds, drop_rate=drop_rate))
    asyncio.create_task(langgraph_response_listener_task())

    loop = asyncio.get_event_loop()

    # Submit N trials (publish initial messages via invoking graph)
    # Note: graph.invoke is synchronous in this compiled graph, so we call it in an executor to avoid blocking the loop.

    # Sequential run
    for i in range(1, n_trials + 1):
        order_id = str(uuid.uuid4())
        initial_state: OrderState = {
            "order_id": order_id,
            "item": item,
            "qty": 2,
            "status": "INIT",
            "inventory_response": None
        }
        conf = {"thread_id": order_id}

        # run synchronously (inside loop), still using thread executor for LLM blocking parts
        result = await loop.run_in_executor(
            None, lambda st=initial_state, conf=conf: app.invoke(st, conf)
        )
        print(f'Result: {result}, \n-----------------')

    # Parallel run
    # sem = asyncio.Semaphore(n_trials)
    #
    # async def run_trial(i):
    #     async with sem:
    #         order_id = str(uuid.uuid4())
    #         initial_state: OrderState = {
    #             "order_id": order_id,
    #             "item": item,
    #             "qty": 2,
    #             "status": "INIT",
    #             "inventory_response": None
    #         }
    #         conf = {"thread_id": order_id}
    #         return await loop.run_in_executor(
    #             None, lambda st=initial_state, conf=conf: app.invoke(st, conf)
    #         )
    # results = await asyncio.gather(*(run_trial(i) for i in range(1, n_trials + 1)))
    # print(results)


    # Let system run for a while so responses are consumed and graph resumed
    await asyncio.sleep(n_trials * 5 + delay_seconds * 2)

    final_stock = get_final_stock(item=item)
    print(f'final_stock is: {final_stock}')

    all_orders = get_all_orders(item=item)
    print(f'all orders:\n')
    print(*all_orders, sep='\n')


# ---- CLI entrypoint ----
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--db-mode", choices=("MOCK", "REAL"), default="REAL")
    p.add_argument("--delay", type=float, default=0.0, help="inventory worker processing delay (sec)")
    p.add_argument("--drop-rate", type=float, default=0.0, help="simulated drop probability (0..1)")
    args = p.parse_args()

    item = "laptop"
    init_stock = 10
    if args.db_mode == 'REAL':
        reset_db(item, init_stock)
        input('Check DB state is clean, press any key to continue ...')

    try:
        asyncio.run(main_runner(n_trials=args.trials, db_mode=args.db_mode,
                                delay_seconds=args.delay, drop_rate=args.drop_rate))
    except KeyboardInterrupt:
        LOG.info("Interrupted - exiting")
