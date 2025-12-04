import asyncio
from typing import TypedDict, Annotated, List, Literal, Optional, Dict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import operator
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from motor.motor_asyncio import AsyncIOMotorClient


# --- Pydantic Models ---

class Order(BaseModel):
    """The order request data, mirroring the intended 'orders' collection entry."""
    order_id: str
    sku_id: str
    quantity: int = Field(ge=1)
    status: Literal = "PENDING"  # Tracks the overall state for the agent


class UpdateResult(BaseModel):
    """Standardized tool output for database operation results."""
    success: bool = Field(description="True if the atomic update succeeded.")
    status_code: Literal
    message: str


class OrderAgentInput(BaseModel):
    """Input schema for the Order Agent's reservation tool call."""
    order_id: str
    sku_id: str
    quantity: int


class InventoryAgentInput(BaseModel):
    """Input schema for the Inventory Agent's confirmation tool call."""
    order_id: str
    action: Literal = Field(description="Must be 'CONFIRM_FULFILLMENT' to proceed.")


class OrderState(TypedDict):
    """The shared state for the LangGraph workflow."""
    order: Order
    messages: Annotated  # Stores LLM prompts and responses
    agent_turn: Literal
    update_result: Optional

# --- MongoDB Client and Service ---

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "supply_chain_db"


class MongoService:
    """Handles asynchronous, atomic interactions with MongoDB using motor."""

    def __init__(self):
        try:
            self.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self.db = self.client
            self.inventory_collection = self.db["inventory"]
            self.orders_collection = self.db["orders"]
        except Exception as e:
            print(f"Error connecting to MongoDB at {MONGO_URL}: {e}")
            raise

    async def initialize_db(self, sku_id: str, initial_stock: int = 10):
        """Initializes the inventory collection atomically if it doesn't exist."""
        await self.inventory_collection.update_one(
            {'sku_id': sku_id},
            {'$set': {'sku_id': sku_id, 'available_stock': initial_stock, 'reserved_stock': 0}},
            upsert=True
        )
        print(f"DB initialized: {sku_id} set to {initial_stock} available.")


MONGO_SERVICE = MongoService()
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1)  # Low temperature for predictability [1]


# --- 3. Tool Definitions (LLM-Callable Functions) ---

@tool("db_reserve_inventory", args_schema=OrderAgentInput, return_direct=False)
async def db_tool_reserve_mongo(order_id: str, sku_id: str, quantity: int) -> UpdateResult:
    """
    Atomically reserves inventory in MongoDB. Fails if stock is insufficient,
    preventing Phantom Consistency race conditions.
    """
    # Key to determinism: Atomic update using $inc and query filter
    # Finds document where available_stock >= quantity AND atomically reserves it.
    result = await MONGO_SERVICE.inventory_collection.update_one(
        # Atomic Filter: Only proceed if stock is sufficient
        {'sku_id': sku_id, 'available_stock': {'$gte': quantity}},
        # Atomic Update: Move stock from available to reserved
        {'$inc': {'available_stock': -quantity, 'reserved_stock': quantity}}
    )

    if result.modified_count == 1:
        # Also update the order status in the orders collection
        await MONGO_SERVICE.orders_collection.update_one(
            {'order_id': order_id},
            {'$set': {'status': 'RESERVED', 'sku_id': sku_id, 'quantity': quantity}},
            upsert=True
        )
        return UpdateResult(success=True, status_code="RESERVED",
                            message=f"Order {order_id} reserved {quantity} units.")
    else:
        # Fails due to stock or race condition (Phantom Consistency failure)
        return UpdateResult(success=False, status_code="INSUFFICIENT_STOCK",
                            message=f"Reservation failed for Order {order_id}. Atomic update lost or stock insufficient.")


@tool("db_confirm_fulfillment", args_schema=InventoryAgentInput, return_direct=False)
async def db_tool_confirm_fulfillment(order_id: str, action: Literal) -> UpdateResult:
    """
    Confirms the order, atomically moving reserved stock to fulfilled/shipped status.
    Requires reserved stock to be present, acting as a final check against Phantom Consistency.
    """
    if action != "CONFIRM_FULFILLMENT":
        return UpdateResult(success=False, status_code="ERROR", message="Invalid action for confirmation.")

    # Fetch order details to get quantity and SKU
    order_doc = await MONGO_SERVICE.orders_collection.find_one({'order_id': order_id})
    if not order_doc:
        return UpdateResult(success=False, status_code="ERROR", message="Order not found in DB.")

    order_quantity = order_doc['quantity']
    sku_id = order_doc['sku_id']

    # Atomic fulfillment: Reduce reserved stock by order_quantity
    result = await MONGO_SERVICE.inventory_collection.update_one(
        # Atomic Filter: Only proceed if reserved stock is sufficient
        {'sku_id': sku_id, 'reserved_stock': {'$gte': order_quantity}},
        # Atomic Update: Reduce reserved stock
        {'$inc': {'reserved_stock': -order_quantity}}
    )

    if result.modified_count == 1:
        # Finalize order status
        await MONGO_SERVICE.orders_collection.update_one(
            {'order_id': order_id},
            {'$set': {'status': 'CONFIRMED'}}
        )
        return UpdateResult(success=True, status_code="ORDER_CONFIRMED",
                            message=f"Order {order_id} fulfillment confirmed.")
    else:
        # Failure suggests reserved stock was not there (Phantom Consistency failure)
        return UpdateResult(success=False, status_code="ERROR",
                            message=f"Confirmation failed for Order {order_id}. Reserved stock mismatch or lost.")


# --- 4. LLM Agent Nodes (Reasoning and Tool Selection) ---

tools = [db_tool_reserve_mongo, db_tool_confirm_fulfillment]


async def order_agent_node(state: OrderState) -> OrderState:
    """Order Agent: Manages reservation initiation and hands off for confirmation."""
    order = state["order"]

    if order.status == "PENDING":
        system_prompt = f"""
        You are the Order Agent. Your goal is to initiate the fulfillment process for Order {order.order_id} (SKU: {order.sku_id}, Qty: {order.quantity}).
        Your current status is PENDING. You MUST use the 'db_reserve_inventory' tool to atomically reserve the required stock.
        """

    elif order.status == "RESERVED":
        # Handoff stage: Order Agent is satisfied and transfers control to Inventory Agent.
        system_prompt = f"""
        You are the Order Agent. Inventory reservation for Order {order.order_id} is complete.
        Your next step is to initiate the final fulfillment confirmation by handing off the task to the Inventory Agent.
        Do not use any tools. State clearly that you are handing off to the Inventory Agent.
        """

    messages = state["messages"] + [{"role": "system", "content": system_prompt}]

    response = await LLM_MODEL.ainvoke(messages)

    new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]

    # Check for tool call (Primary goal: Reservation)
    if response.tool_calls:
        tool_call = response.tool_calls

        # LLM Non-determinism Test: Tool argument generation must adhere to Pydantic schema
        tool_args = tool_call.args
        tool_function = next(t for t in tools if t.name == tool_call.name)
        result = await tool_function(**tool_args)

        new_messages.append({"role": "tool_result", "content": str(result.dict())})

        return {
            "messages": new_messages,
            "agent_turn": "ROUTER",
            "update_result": result,
        }

    # Handoff logic (Secondary goal: When status is RESERVED)
    if order.status == "RESERVED":
        print(f"[Order Agent Handoff]: {response.content}")
        return {
            "messages": new_messages,
            "agent_turn": "INVENTORY",
        }

    return state  # Should not happen in fixed flow


async def inventory_agent_node(state: OrderState) -> OrderState:
    """Inventory Agent: Finalizes fulfillment by confirming the transaction."""
    order = state["order"]

    system_prompt = f"""
    You are the Inventory Agent. The Order Agent has successfully reserved stock for Order {order.order_id}.
    Your task is to finalize the transaction. You MUST use the 'db_confirm_fulfillment' tool now with the action 'CONFIRM_FULFILLMENT' 
    to atomically finalize the stock movement in MongoDB.
    """

    messages = state["messages"] + [{"role": "system", "content": system_prompt}]

    response = await LLM_MODEL.ainvoke(messages)

    new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]

    if response.tool_calls:
        tool_call = response.tool_calls
        tool_function = next(t for t in tools if t.name == tool_call.name)
        tool_args = tool_call.args
        result = await tool_function(**tool_args)

        new_messages.append({"role": "tool_result", "content": str(result.dict())})

        return {
            "messages": new_messages,
            "agent_turn": "ROUTER",
            "update_result": result,
        }

    return state


# --- 5. Conditional Routing Logic (Enforcing Protocol Consistency) ---

def router_node(state: OrderState) -> str:
    """
    Deterministic Router: Enforces the procedural protocol based on the tool's result.
    This prevents non-deterministic LLM reasoning from breaking the transaction flow.
    """
    update_result = state["update_result"]
    order = state["order"]

    if update_result.success:
        if update_result.status_code == "RESERVED":
            # Reservation successful: update local state and transition to Order Agent (for handoff decision)
            order.status = "RESERVED"
            print(f": ROUTER: Reservation successful. Next: Order Agent handoff.")
            return "ORDER"

        elif update_result.status_code == "ORDER_CONFIRMED":
            # Final fulfillment successful: end.
            order.status = "CONFIRMED"
            print(f": ROUTER: Fulfillment Confirmed. Ending workflow.")
            return "END"

    else:  # Failure
        # Transactional failure due to stock or Phantom Consistency loss.
        order.status = "CANCELLED"
        print(f": ROUTER: Transaction failed ({update_result.status_code}). Ending workflow.")
        return "END"


# --- 6. LangGraph Definition and Execution ---

def build_graph():
    """Defines the multi-agent workflow using LangGraph."""
    workflow = StateGraph(OrderState)

    workflow.add_node("ORDER", order_agent_node)
    workflow.add_node("INVENTORY", inventory_agent_node)
    workflow.add_node("ROUTER", router_node)

    workflow.set_entry_point("ORDER")

    workflow.add_edge("ORDER", "ROUTER")
    workflow.add_edge("INVENTORY", "ROUTER")

    workflow.add_conditional_edges(
        "ROUTER",
        router_node,
        {
            "ORDER": "ORDER",  # Reservation success, loop back to Order Agent
            "INVENTORY": "INVENTORY",  # Handoff from Order Agent to Inventory Agent
            "END": END  # Failure or final success
        }
    )

    return workflow.compile()


# --- Execution Simulation ---

compiled_graph = build_graph()


async def get_db_status(sku: str) -> Dict:
    """Utility to fetch the current inventory status."""
    doc = await MONGO_SERVICE.inventory_collection.find_one({'sku_id': sku}, {'_id': 0})
    return doc if doc else {"message": "SKU not found"}


async def run_experiment(order_id: str, sku: str, quantity: int, description: str):
    print(f"\n{'=' * 80}")
    print(f"SIMULATION: {description}")
    print(f"{'=' * 80}")

    # Reset DB state for this run (ensure A101 is available)
    await MONGO_SERVICE.initialize_db(sku_id="A101", initial_stock=10)

    initial_state = Order(order_id=order_id, sku_id=sku, quantity=quantity)
    initial_graph_state = {
        "order": initial_state,
        "messages": [{"role": "user", "content": f"Fulfill order {order_id} for {quantity} units of {sku}."}],
        "agent_turn": "ORDER",
        "update_result": None
    }

    final_state = await compiled_graph.ainvoke(initial_graph_state)
    final_inventory = await get_db_status(sku)

    print(f"\n{'*' * 80}")
    print(f"FINAL RESULT FOR ORDER {order_id}: {final_state['order'].status}")
    print(f"MongoDB Inventory Status ({sku}): {final_inventory}")
    print(f"{'*' * 80}\n")


async def main():
    # --- Scenario 1: Successful Atomic Reservation and Confirmation (3 units) ---
    # Expected Outcome: CONFIRMED. Available: 7, Reserved: 0.
    await run_experiment(
        order_id="ORD-001",
        sku="A101",
        quantity=3,
        description="Scenario 1: Successful Full-Cycle Transaction (Expected: CONFIRMED)"
    )

    # --- Scenario 2: Insufficient Stock (Testing Failure Protocol) ---
    # We re-initialize the DB to 10 stock. This order requires 20 units.
    # Expected Outcome: CANCELLED. Available: 10, Reserved: 0.
    await run_experiment(
        order_id="ORD-002",
        sku="A101",
        quantity=20,
        description="Scenario 2: Attempted Order with Insufficient Stock (Expected: CANCELLED)"
    )


if __name__ == "__main__":
    # Initialize the DB structure before starting the main async loop
    asyncio.run(main())