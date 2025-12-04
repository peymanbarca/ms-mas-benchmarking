# fulfillment_system.py (Run this last)
import asyncio
import httpx
from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import operator
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# --- Configuration ---
ORDER_SERVICE_URL = "http://localhost:8001"
INVENTORY_SERVICE_URL = "http://localhost:8000"
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1)


# --- Shared Pydantic Models ---
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
    order: Order
    messages: Annotated[List[str], operator.add]
    last_status_code: Optional[str]
    turn: Literal


# --- Tool Definition (Shared Microservice Tool) ---

class MicroserviceToolInput(BaseModel):
    service: Literal
    action: Literal
    order_id: str
    quantity: int = Field(default=0, description="Required only for inventory actions or initial order creation.")


@tool("microservice_action", args_schema=MicroserviceToolInput, return_direct=False)
async def microservice_tool(service: Literal, action: Literal, order_id: str, quantity: int = 0) -> UpdateResult:
    """Dispatches asynchronous calls to the correct microservice (Order or Inventory)."""
    client = httpx.AsyncClient()

    # Determine URL and Data based on Service and Action (Protocol)
    if service == "ORDER":
        url = f"{ORDER_SERVICE_URL}/orders/"
        data = {"order_id": order_id}
        if action == "CREATE":
            url += "create"
            data.update({"quantity": quantity, "sku_id": "A101"})
            method = 'POST'
        elif action == "UPDATE_STATUS":
            url += "update_status"
            data["status"] = quantity  # Misusing quantity field to pass status
            method = 'POST'
    elif service == "INVENTORY":
        url = f"{INVENTORY_SERVICE_URL}/inventory/"
        data = {"order_id": order_id}
        if action == "RESERVE":
            url += "reserve"
            data["quantity"] = quantity
            method = 'PUT'
        elif action == "CONFIRM":
            url += "confirm_deduct"
            data["quantity"] = quantity  # Need quantity for confirmation logic
            method = 'POST'
        elif action == "COMPENSATE":
            url += "compensate"
            data["quantity"] = quantity
            method = 'POST'

    # Execute the asynchronous call
    try:
        response = await client.request(method, url, json=data)
        response.raise_for_status()
        return UpdateResult.parse_obj(response.json())
    except Exception as e:
        return UpdateResult(success=False, status_code="EXTERNAL_ERROR", message=f"Service failure: {e}")
    finally:
        await client.aclose()


# --- Agent Nodes (LLM Reasoning) ---

tools = [microservice_tool]


async def order_agent_node(state: AgentState) -> AgentState:
    """Order Agent: Creates order, reserves stock, and updates status."""
    order = state["order"]

    if order.status == "PENDING":
        # Step 1: Create Order Record (Local transaction in Order Service)
        action = "CREATE"
        service = "ORDER"
        quantity = order.quantity
    elif state["last_status_code"] == "CREATED" or order.status == "CREATED":
        # Step 2: Reserve Inventory
        action = "RESERVE"
        service = "INVENTORY"
        quantity = order.quantity
    elif order.status == "RESERVED":
        # Step 3: Handoff
        return {"messages": ["Handoff to Inventory Agent."], "turn": "INVENTORY", "last_status_code": "HANDOFF"}
    else:
        return state  # No action

    system_prompt = f"You are the Order Agent. Current Status: {order.status}. You must call 'microservice_action' to {action} on the {service} service."

    # LLM execution logic (simplified to focus on tool call)
    result: UpdateResult = await microservice_tool(
        service=service, action=action, order_id=order.order_id, quantity=quantity
    )

    new_status = result.status_code if result.success else f"FAIL_{action}"

    # If reservation was successful, update local state
    if new_status == "RESERVED":
        state["order"].status = "RESERVED"

    return {
        "messages": [f"[Order Agent Action]: {action} on {service}. Status: {new_status}"],
        "last_status_code": new_status,
        "order": state["order"],
        "turn": "ROUTER"
    }


async def inventory_agent_node(state: AgentState) -> AgentState:
    """Inventory Agent: Confirms fulfillment or initiates compensation/rollback."""
    order = state["order"]

    if order.status == "RESERVED":
        # Step 4: Confirm Inventory Deduction
        action = "CONFIRM"
    elif state["last_status_code"] in:
        # Step 5: Compensation (Rollback Protocol)
        action = "COMPENSATE"
    else:
        return state

    system_prompt = f"You are the Inventory Agent. Current Status: {order.status}. You must call 'microservice_action' to {action} on the INVENTORY service."

    # LLM execution logic
    result: UpdateResult = await microservice_tool(
        service="INVENTORY", action=action, order_id=order.order_id, quantity=order.quantity
    )

    new_status = result.status_code if result.success else f"FAIL_{action}"

    # Update Order Service status as checkpoint
    if new_status == "DEDUCTED":
        await microservice_tool(service="ORDER", action="UPDATE_STATUS", order_id=order.order_id, quantity="FULFILLED")

    return {
        "messages": [f"[Inventory Agent Action]: {action}. Status: {new_status}"],
        "last_status_code": new_status,
        "order": order.copy(),
        "turn": "ROUTER"
    }


# --- 6. Conditional Routing Logic (Protocol Enforcement) ---

def router_node(state: AgentState) -> str:
    status = state["last_status_code"]

    # Termination conditions
    if status in:
        return "END"

    # Progression: Hand off to Inventory Agent
    if status == "RESERVED":
        return "INVENTORY"

    # Failure condition: Reservation failed, initiate compensation/cancellation protocol
    if status in:
        return "INVENTORY"  # Route to Inventory Agent for COMPENSATE logic

    # Default: Loop back to Order Agent to continue or retry
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

    # Router logic
    workflow.add_conditional_edges(
        "ROUTER",
        router_node,
        {"ORDER": "ORDER", "INVENTORY": "INVENTORY", "END": END}
    )
    return workflow.compile()


# --- Execution Simulation ---

async def main():
    # Run the microservices first: inventory_service (8000) & order_service (8001)

    compiled_graph = build_graph()

    async def run_scenario(order_id, quantity, description):
        # Initial call to ensure DB is clean for the test
        await microservice_tool(service="ORDER", action="CREATE", order_id=order_id, quantity=quantity)

        initial_order = Order(order_id=order_id, sku_id="A101", quantity=quantity)
        initial_graph_state = {
            "order": initial_order,
            "messages": [f"User: Fulfill order {order_id} for {quantity} units of A101."],
            "last_status_code": "CREATED",
            "turn": "ORDER"
        }

        final_state = await compiled_graph.ainvoke(initial_graph_state)
        print(f"\nFINAL STATUS for {order_id}: {final_state['last_status_code']}")

    # Scenario 1: Successful Distributed Transaction (Expected: FULFILLED)
    await run_scenario("ORD-2A2M-001", 3, "Successful Distributed Transaction (Reserve -> Deduct)")

    # Scenario 2: Failure in Step 2 triggers Compensation (Expected: COMPENSATED)
    # The Order Agent will try to RESERVE, Microservice will return INSUFFICIENT_STOCK
    await run_scenario("ORD-2A2M-002", 100, "Failure: Insufficient Stock triggers COMPENSATION SAGA")


if __name__ == "__main__":
    asyncio.run(main())