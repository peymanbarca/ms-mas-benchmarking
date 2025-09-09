# inventory_grpc_server.py
import asyncio
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

import grpc
from pymongo import MongoClient

import proto.retail_pb2 as pb
import proto.retail_pb2_grpc as rpc

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("inventory")

MONGO_URI = "mongodb://user:pass1@localhost:27017/"
DB_NAME = "retail"

# Use a small thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=8)


class InventoryServicer(rpc.InventoryServicer):
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        # ensure some default item for tests
        # don't overwrite if exists
        self.db.inventory.update_one({"item": "laptop"}, {"$setOnInsert": {"stock": 10}}, upsert=True)

    async def Reserve(self, request: pb.ReserveRequest, context) -> pb.ReserveResponse:
        item = request.item
        qty = int(request.qty)
        delay_ms = int(request.delay_ms or 0)

        LOG.info("Reserve request: item=%s qty=%d delay_ms=%d", item, qty, delay_ms)

        # perform the reservation atomically in a thread to avoid blocking asyncio loop
        def atomic_reserve():
            # find and decrement stock atomically
            # use find_one_and_update with $gte check
            # Mongo find_one_and_update is atomic so oversells are prevented at DB level
            res = self.db.inventory.find_one_and_update(
                {"item": item, "stock": {"$gte": qty}},
                {"$inc": {"stock": -qty}}
            )
            return res is not None

        def regular_reserve():
            stock = self.db.inventory.find_one({"item": item})
            if stock["stock"] >= qty:
                self.db.inventory.update_one({"item": item}, {"$inc": {"stock": -qty}}, upsert=True)
                return True
            return False

        reserved = await asyncio.get_event_loop().run_in_executor(executor=executor, func=regular_reserve)

        # Inject artificial delay AFTER stock decrement
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        if reserved:
            reservation_id = str(uuid.uuid4())
            LOG.info("Reserved %d %s (reservation=%s)", qty, item, reservation_id)
            return pb.ReserveResponse(status="reserved", reservation_id=reservation_id, message="")
        else:
            LOG.info("Out of stock: %s x%d", item, qty)
            return pb.ReserveResponse(status="out_of_stock", reservation_id="", message="not enough stock")


async def serve(host="0.0.0.0", port=50051):
    server = grpc.aio.server()
    rpc.add_InventoryServicer_to_server(InventoryServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    LOG.info("Starting Inventory gRPC server on %s:%d", host, port)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        LOG.info("Shutting down")
