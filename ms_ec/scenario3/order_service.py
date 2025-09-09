# order_service.py
import uuid
import logging
import asyncio
from fastapi import FastAPI
from pymongo import MongoClient
import ms_ec.scenario3.proto.retail_pb2 as pb
import ms_ec.scenario3.proto.retail_pb2_grpc as rpc
import grpc

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("order")

app = FastAPI(title="Order Service")

MONGO_URI = "mongodb://user:pass1@localhost:27017/"
DB_NAME = "retail"
INVENTORY_ADDR = "localhost:50051"  # gRPC host:port

# Mongo client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Create gRPC channel and stub on startup to reuse
grpc_channel = None
inventory_stub = None

@app.on_event("startup")
async def startup_event():
    global grpc_channel, inventory_stub
    # Use insecure channel for local testing — replace with secure creds in prod
    grpc_channel = grpc.aio.insecure_channel(INVENTORY_ADDR)
    inventory_stub = rpc.InventoryStub(grpc_channel)
    # Optionally wait for channel ready (short timeout)
    try:
        await grpc_channel.channel_ready()
        LOG.info("Connected to Inventory gRPC at %s", INVENTORY_ADDR)
    except Exception:
        LOG.warning("gRPC channel not ready immediately")

@app.on_event("shutdown")
async def shutdown_event():
    global grpc_channel
    if grpc_channel is not None:
        await grpc_channel.close()

@app.post("/clear_orders")
def clear_orders():
    db.orders.delete_many({})

@app.post("/order")
async def create_order(request: dict):
    item: str = request["item"]
    qty: int = int(request["qty"])
    delay_ms: int = int(request.get("delay_ms", 0))
    order_id = str(uuid.uuid4())

    # Persist INIT order
    db.orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT"})

    # Build gRPC request
    grpc_req = pb.ReserveRequest(item=item, qty=qty, delay_ms=delay_ms)

    try:
        # Call gRPC Inventory service (async)
        # create_order is async and uses the inventory_stub.Reserve async stub call.
        res: pb.ReserveResponse = await inventory_stub.Reserve(grpc_req, timeout=10.0)
    except grpc.aio.AioRpcError as e:
        LOG.exception("gRPC call failed: %s", e)
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED"}})
        return {"order_id": order_id, "final_status": "FAILED", "error": str(e)}

    # Handle response
    if res.status == "reserved":
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "RESERVED"}})
        # Mock payment success immediately
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "COMPLETED"}})
    elif res.status == "out_of_stock":
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED_OUT_OF_STOCK"}})
    else:
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED"}})

    final = db.orders.find_one({"_id": order_id})
    return {"order_id": order_id, "final_status": final["status"]}
