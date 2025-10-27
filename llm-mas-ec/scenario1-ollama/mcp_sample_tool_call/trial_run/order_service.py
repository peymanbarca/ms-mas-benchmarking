import time

from fastapi import FastAPI
from pymongo import MongoClient
import requests, uuid
from pydantic import BaseModel
import json
from typing import Dict, Any

app = FastAPI(title="Order Service")
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]

# --- CONFIGURATION ---
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
TOOL_NAME = "reserve_stock"
ORDER_SERVICE_PORT = 8001


# We cannot import fastmcp.Client here because of the file generation constraint
# We assume fastmcp.Client is imported and available in the environment

class OrderRequest(BaseModel):
    item: str
    qty: int


# Initialize FastAPI app
app = FastAPI(
    title="Order Service",
    description="Handles order placement and uses FastMCP for inventory."
)

# Initialize the MCP client globally
app.state.mcp_client = None


# Internal order creation and reservation logic (extracted from the endpoint)
async def create_order_internal(item: str, qty: int) -> Dict[
    str, Any]:
    client = app.state.mcp_client

    order_id = str(uuid.uuid4())
    db.orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT"})

    # 1. Call FastMCP Client asynchronously
    try:
        # The client automatically serializes the request and deserializes the JSON-RPC response
        t1 = time.time()
        reservation_result = await client.call_tool(
            name=TOOL_NAME,
            arguments={
                "order_id": order_id,
                "item": item,
                "qty": qty
            }
        )
        t2 = time.time()
        print(f'Tool Call took: {round((t2-t1),3)}')

        # FastMCP returns a Result object; the server's return value is in .data
        response = json.loads(reservation_result.content[0].text)
        res = {
              "jsonrpc": "2.0",
              "id": response["order_id"],
              "data": response
        }
        # print(json.dumps(res))

        with open('mcp_parallel_call.txt', 'a') as f1:
            f1.write(f'Total Tool Call Response Took: {round((t2-t1), 3)}\n')

        if res["data"]["status"] == "reserved":
            db.orders.update_one({"_id": order_id}, {"$set": {"status": "RESERVED"}})
            # Mock payment success
            db.orders.update_one({"_id": order_id}, {"$set": {"status": "COMPLETED"}})
        elif res["data"]["status"] == "out_of_stock":
            db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED_OUT_OF_STOCK"}})
        else:
            db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED"}})

        return {"order_id": order_id, "final_status": db.orders.find_one({"_id": order_id})["status"],
                "reservation_id": str(uuid.uuid4())}

    except Exception as e:
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED"}})


@app.on_event("startup")
async def startup_event():
    """Initializes the FastMCP Client and sets up the async context."""
    from fastmcp import Client  # Import client here to ensure it's loaded properly
    print(f"Connecting FastMCP client to: {MCP_SERVER_URL}")

    # Initialize the client (it will handle the HTTP/SSE transport)
    client = Client(MCP_SERVER_URL)

    # Start the client's internal session manager
    await client.__aenter__()
    app.state.mcp_client = client


@app.on_event("shutdown")
async def shutdown_event():
    """Shuts down the FastMCP Client session."""
    client = app.state.mcp_client
    if client:
        await client.__aexit__(None, None, None)
        print("FastMCP client session closed.")


@app.post("/order")
async def create_order_endpoint(request: OrderRequest):
    """
    Asynchronous endpoint to create a single order and reserve inventory via MCP.
    """
    result = await create_order_internal(
        item=request.item,
        qty=request.qty
    )
    return result