import asyncio
from mcp.server.fastmcp import FastMCP, Context
from pymongo import MongoClient

# DB client
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail_mas_mcp"]
inventory_col = db.inventory

mcp = FastMCP(name="inventory_mcp_server")

# Define the host and port for the HTTP/SSE server
HOST = "127.0.0.1"
PORT = 8000

@mcp.tool()
def reserve_stock(order_id: str, item: str, qty: int) -> dict:
    # print(f'New Request: {order_id}, {item}, {qty}')
    stock_doc = inventory_col.find_one({"item": item})
    if not stock_doc:
        # stock_doc = {"item": item, "stock": 10}
        # inventory_col.insert_one(stock_doc)
        return {"order_id": order_id, "status": "out_of_stock"}

    if stock_doc["stock"] >= qty:
        inventory_col.update_one({"item": item}, {"$inc": {"stock": -qty}}, upsert=True)
        return {"order_id": order_id, "status": "reserved"}
    else:
        return {"order_id": order_id, "status": "out_of_stock"}


if __name__ == "__main__":

    # mcp.run(transport="stdio")

    print(f"Starting FastMCP server on http://{HOST}:{PORT}/mcp ...")
    # Crucial change: Explicitly set transport to 'http' and provide host/port.
    # The client will connect to this address.
    mcp.run(transport='streamable-http', mount_path=f'{HOST}:{PORT}')
