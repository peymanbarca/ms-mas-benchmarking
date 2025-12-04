# fulfillment_system.py (Run this file second)
import asyncio
import httpx
from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import operator
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# --- Configuration ---
MICROSERVICE_URL = "http://localhost:8000"
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1)


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
    messages: Annotated[List[str], operator.add]
    last_status_code: Optional[str]
    turn: Literal  # Tracks which agent should execute next


# --- Tool Definition (Shared Microservice Wrapper) ---

class MicroserviceToolInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the order.")
    action: Literal
    quantity: int = Field(default=0, description="Required only for the RESERVE action.")


@tool("microservice_action", args_schema=MicroserviceToolInput, return_direct=False)
async def microservice_tool(order_id: str, action: Literal, quantity: int = 0) -> UpdateResult:
    """
    A single tool used by both agents to interact with the Inventory Microservice (FastAPI).
    Action must be 'RESERVE', 'CONFIRM', or 'COMPENSATE'.
    """
    client = httpx.AsyncClient()
    url = f"{MICROSERVICE_URL}/inventory/"
    data = {"order_id": order_id, "action": action}

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
            return UpdateResult(success=False, status_code="INVALID_ACTION", message=f"Invalid action: {action}")

        response.raise_for_status()
        return UpdateResult.parse_obj(response.json())

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP Error {e.response.status_code}: Pydantic validation failure or server error."
        return UpdateResult(success=False, status_code="LLM_ARG_ERROR", message=error_msg)

    except Exception as e:
        return UpdateResult(success=False, status_code="NETWORK_ERROR", message=f"Network or connection error: {e}")
    finally:
        await client.aclose()


# --- Agent Nodes (LLM Reasoning) ---

tools = [microservice_tool]


async def order_agent_node(state: AgentState) -> AgentState:
    """Order Agent: Focuses on initial reservation and ensuring resource locking."""
    order = state["order"]

    # Order Agent must decide between RESERVE (if PENDING) or HANDOFF (if RESERVED)
    if order.status == "PENDING" or state["last_status_code"] == "START":
        action = "RESERVE"
    else:
        # If already reserved, Order Agent's job is done; it confirms the handoff.
        action = "HANDOFF"

    system_prompt = f"""
    You are the Order Agent. Current Status: {order.status}. Last Result: {state['last_status_code']}.
    If the status is PENDING, you MUST call the 'microservice_action' tool with action='RESERVE' and the order quantity.
    If the status is RESERVED, do not call any tools; explicitly state you are handing off to the Inventory Agent.
    """

    messages = state["messages"] + [{"role": "system", "content": system_prompt}]
    response = await LLM_MODEL.ainvoke(messages)
    new_messages = state["messages"] + [response.content]

    if action == "RESERVE" and response.tool_calls:
        tool_args = response.tool_calls.args
        result: UpdateResult = await microservice_tool(
            order_id=order.order_id,
            action=tool_args.get("action"),
            quantity=order.quantity
        )
        new_messages.append(f": {result.status_code}")

        return {
            "messages": new_messages,
            "last_status_code": result.status_code,
            "order": order.copy(),
            "turn": "ROUTER"
        }

    # Handoff logic
    if action == "HANDOFF" and order.status == "RESERVED":
        print(f"[Order Agent]: Handoff to Inventory Agent for final confirmation.")
        return {
            "messages": new_messages,
            "last_status_code": "HANDOFF_TO_INVENTORY",
            "order": order.copy(),
            "turn": "INVENTORY"  # Direct transition to Inventory Agent
        }

    # If PENDING but failed to call tool (LLM inconsistency)
    return {
        "messages": new_messages + ["Order Agent failed to execute tool call."],
        "last_status_code": "ORDER_AGENT_FAILURE",
        "order": order.copy(),
        "turn": "ROUTER"
    }


async def inventory_agent_node(state: AgentState) -> AgentState:
    """Inventory Agent: Focuses on confirming fulfillment or initiating compensation."""
    order = state["order"]

    system_prompt = f"""
    You are the Inventory Agent. Current Status: {order.status}. Last Result: {state['last_status_code']}.
    Your task is critical: If the status is RESERVED, you MUST call 'microservice_action' with action='CONFIRM'.
    If the status is INSUFFICIENT_STOCK or any prior status indicates failure, you MUST call 'microservice_action' with action='COMPENSATE' to rollback the order.
    """

    messages = state["messages"] + [{"role": "system", "content": system_prompt}]
    response = await LLM_MODEL.ainvoke(messages)
    new_messages = state["messages"] + [response.content]

    # Assume the agent correctly decides CONFIRM or COMPENSATE
    if order.status == "RESERVED":
        action = "CONFIRM"
    else:
        action = "COMPENSATE"  # Trigger rollback protocol

    if response.tool_calls:
        tool_args = response.tool_calls.args

        # Use the decided action, overriding potential LLM error in tool arg generation
        result: UpdateResult = await microservice_tool(
            order_id=order.order_id,
            action=action,  # Enforce the correct action based on graph logic
        )
        new_messages.append(f": {result.status_code}")

        return {
            "messages": new_messages,
            "last_status_code": result.status_code,
            "order": order.copy(),
            "turn": "ROUTER"
        }

    return state  # Should not happen


# --- 6. Conditional Routing Logic (Protocol Enforcement) ---

def router_node(state: AgentState) -> str:
    """
    Deterministic Router: Manages the high-level workflow transitions.
    """
    status = state["last_status_code"]

    # 1. Termination conditions
    if status in:
        state["order"].status = status  # Update final status
        return "END"

    # 2. Handoff from Order Agent to Inventory Agent
    if status == "RESERVED":
        # The agent's turn must now shift from ORDER to INVENTORY
        return "INVENTORY"

    # 3. Handle failure/compensation state
    if status in:
        # If reservation fails, route to Inventory Agent to initiate COMPENSATE
        return "INVENTORY"

    # 4. Loop back (e.g., if Order Agent needs to try again or if status is ambiguous)
    return "ORDER"


# --- 7. LangGraph Definition and Execution ---

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("ORDER", order_agent_node)
    workflow.add_node("INVENTORY", inventory_agent_node)
    workflow.add_node("ROUTER", router_node)

    workflow.set_entry_point("ORDER")

    # Transitions after agents run
    workflow.add_edge("ORDER", "ROUTER")
    workflow.add_edge("INVENTORY", "ROUTER")

    # Router logic (Conditional edges)
    workflow.add_conditional_edges(
        "ROUTER",
        router_node,
        {
            "ORDER": "ORDER",  # Loop back to Order Agent (e.g., failed to call tool, retry)
            "INVENTORY": "INVENTORY",  # Handoff or Compensation
            "END": END  # Terminal state
        }
    )
    return workflow.compile()


# --- Execution Simulation ---
async def main():
    # NOTE: You MUST run inventory_microservice.py first on port 8000

    compiled_graph = build_graph()

    async def run_scenario(order_id, quantity, description):
        print(f"\n{'=' * 80}\nSCENARIO: {description}\n{'=' * 80}")
        initial_order = Order(order_id=order_id, sku_id="A101", quantity=quantity)
        initial_graph_state = {
            "order": initial_order,
            "messages": [f"User: Fulfill order {order_id} for {quantity} units of A101."],
            "last_status_code": "START",
            "turn": "ORDER"
        }

        final_state = await compiled_graph.ainvoke(initial_graph_state)
        print(f"\nFINAL STATUS for {order_id}: {final_state['order'].status}")

    # Scenario 1: Successful Coordination
    # Expected: Order Agent reserves, Router hands off, Inventory Agent confirms.
    await run_scenario("ORD-2A-001", 3, "Successful Reserve and Confirm (Coordination Test)")

    # Scenario 2: Failure and Compensation Protocol Test
    # We assume the microservice has 10 stock, and this one requires 100.
    # Expected: Order Agent RESERVE fails (INSUFFICIENT_STOCK), Router sends to Inventory Agent, Inventory Agent compensates/cancels.
    await run_scenario("ORD-2A-002", 100, "Failure Test: Insufficient Stock triggers Compensation Protocol")


if __name__ == "__main__":
    asyncio.run(main())