# fulfillment_agent.py
import asyncio
import httpx  # Asynchronous HTTP client for calling FastAPI
from typing import TypedDict, Annotated, List, Literal, Optional, Dict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import operator
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# --- Configuration ---
MICROSERVICE_URL = "http://localhost:8000"
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1)  # Low T for high consistency


# --- Pydantic Models (Shared Contracts) ---
class Order(BaseModel):
    order_id: str
    sku_id: str
    quantity: int
    status: str = "PENDING"


class UpdateResult(BaseModel):
    success: bool
    status_code: str
    message: str


class AgentState(TypedDict):
    """The shared state for the LangGraph workflow."""
    order: Order
    messages: Annotated
    last_status_code: Optional[str]


# --- Tool Definition (LLM-Callable Microservice Wrapper) ---

# Note: The LLM generates arguments for the tool, which then constructs the API call.
@ tool("microservice_action", args_schema=ReservationRequest, return_direct=False)
async def microservice_tool(order_id: str, quantity: int, action: Literal) -> UpdateResult:
    """
    Interacts with the Inventory Microservice (FastAPI) to perform a transactional action.
    Use RESERVE to reserve stock. Use CONFIRM to confirm fulfillment. Use COMPENSATE for rollback.
    """
    client = httpx.AsyncClient()
    url = f"{MICROSERVICE_URL}/inventory/"
    data = {"order_id": order_id}

    try:
        if action == "RESERVE":
            url += "reserve"
            data["quantity"] = quantity
            response = await client.put(url, json=data)
        elif action == "CONFIRM":
            url += "confirm"
            response = await client.post(url, json=data)
        elif action == "COMPENSATE":
            url += "compensate"
            response = await client.post(url, json=data)
        else:
            return UpdateResult(success=False, status_code="ERROR", message=f"Invalid action: {action}")

        # Pydantic validation of the deterministic microservice response
        response.raise_for_status()
        result_data = response.json()
        return UpdateResult.parse_obj(result_data)

    except httpx.HTTPStatusError as e:
        # This catches errors like HTTP 422 (Pydantic validation failure in the microservice)
        error_msg = f"HTTP Error {e.response.status_code}: LLM arguments failed Pydantic validation."
        print(f" -- TOOL FAILED: {error_msg}")
        return UpdateResult(success=False, status_code="LLM_ARG_ERROR", message=error_msg)

    except Exception as e:
        return UpdateResult(success=False, status_code="ERROR", message=f"Tool execution failed: {e}")
    finally:
        await client.aclose()

# --- 5. Single Agent Node (LLM Reasoning) ---

async def fulfillment_agent_node(state: AgentState) -> AgentState:
    """
    The single agent that reasons and calls the microservice tool in a loop.
    This tests the agent's Planning/Intention persistence.
    """
    order = state["order"]

    system_prompt = f"""
You are the Fulfillment Agent. Your mission is to ensure Order {order.order_id} (Qty: {order.quantity}) is confirmed.
Current Order Status: {order.status}. Last Microservice Result: {state['last_status_code']}.

Follow this rigid protocol:
1. If status is PENDING, call the 'microservice_action' tool with 'RESERVE'.
2. If status is RESERVED, call the 'microservice_action' tool with 'CONFIRM'.
3. If any step fails (e.g., INSUFFICIENT_STOCK), you MUST call 'microservice_action' with 'COMPENSATE' to rollback, and then exit.

You must always output a tool call until the order status is CONFIRMED or COMPENSATED/CANCELLED.
"""

    messages = state["messages"] + [{"role": "system", "content": system_prompt}]

    response = await LLM_MODEL.ainvoke(messages)

    new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]

    # Check for tool call
    if response.tool_calls:
        tool_call = response.tool_calls  # Assume one call per turn

        # Non-deterministic tool argument generation is tested here.
        tool_args = tool_call.args
        result: UpdateResult = await microservice_tool(
            order_id=tool_args.get("order_id"),
            quantity=tool_args.get("quantity", 0),  # Pass 0 if quantity is not needed for confirm/compensate
            action=tool_args.get("action")
        )

        new_messages.append({"role": "tool_result", "content": str(result.dict())})

        return {
            "messages": new_messages,
            "last_status_code": result.status_code,
            "order": order.copy(update={"status": result.status_code}) if result.success else order.copy(),
        }

    # If LLM fails to call tool or attempts to exit prematurely
    print(f"WARNING: LLM failed to output tool call. Content: {response.content}")
    return {
        "messages": new_messages,
        "last_status_code": "ERROR",
        "order": order.copy(update={"status": "ERROR"}),
    }

# --- 6. Conditional Routing Logic (Protocol Enforcement) ---

def router_node(state: AgentState) -> str:
    """
    Deterministic Router: Enforces termination based on the microservice's final status.
    This prevents the LLM from hallucinating an exit or getting stuck in a loop.
    """
    status = state["last_status_code"]

    if status in:
        return "END"

    # Continue to the agent for the next step (Reserve -> Confirm)
    return "AGENT"

# --- 7. LangGraph Definition and Execution ---

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("AGENT", fulfillment_agent_node)
    workflow.add_node("ROUTER", router_node)

    workflow.set_entry_point("AGENT")

    workflow.add_edge("AGENT", "ROUTER")

    workflow.add_conditional_edges(
        "ROUTER",
        router_node,
        {"AGENT": "AGENT", "END": END}
    )
    return workflow.compile()

# --- Execution Simulation ---

async def run_experiment(order_id: str, quantity: int, description: str):
    print(f"\n{'=' * 80}")
    print(f"SIMULATION: {description}")
    print(f"{'=' * 80}")

    initial_order = Order(order_id=order_id, sku_id="A101", quantity=quantity)
    initial_graph_state = {
        "order": initial_order,
        "messages": [{"role": "user", "content": f"Fulfill order {order_id} for {quantity} units of A101."}],
        "last_status_code": "START",
    }

    final_state = await build_graph().ainvoke(initial_graph_state)

    print(f"\n{'*' * 80}")
    print(f"FINAL RESULT FOR ORDER {order_id}: {final_state['order'].status}")
    print(f"{'*' * 80}\n")

# To run: Execute the microservice first, then run this file:
# if __name__ == "__main__":
#     # NOTE: The microservice must be running on localhost:8000
#     asyncio.run(run_experiment(
#         order_id="ORD-EX2-001",
#         quantity=4,
#         description="Successful Reserve and Confirm"
#     ))
#     asyncio.run(run_experiment(
#         order_id="ORD-EX2-002",
#         quantity=100,
#         description="Failure (Insufficient Stock) and Rollback Protocol Test"
    #     ))