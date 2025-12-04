import os
import time
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient, ReturnDocument

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "retail_exp2")
COLL_NAME = os.environ.get("COLL_NAME", "inventory")

# Fault injection via env
INJECT_DELAY = float(os.environ.get("SERVICE_DELAY", "0"))   # seconds
INJECT_DROP_RATE = int(os.environ.get("SERVICE_DROP_RATE", "0"))  # percentage 0-100

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
inventory_col = db[COLL_NAME]

app = FastAPI(title="Inventory Microservice")

class ReserveReq(BaseModel):
    item: str
    qty: int
    request_id: str | None = None
    atomic_update: bool = False

@app.on_event("startup")
def startup():
    # ensure collection exists
    inventory_col.create_index("item", unique=True)

@app.post("/reserve")
def reserve(req: ReserveReq):
    """
    Try to reserve qty of item atomically.
    Response: {"reserved": True/False, "item": ..., "remaining": n}
    """
    # optional drop injection: simulate network failure by returning 500 occasionally
    if INJECT_DROP_RATE > 0:
        import random
        if random.randint(0, 99) < INJECT_DROP_RATE:
            # simulate dropped request
            raise HTTPException(status_code=503, detail="simulated service drop")

    # inject delay
    if INJECT_DELAY > 0:
        time.sleep(INJECT_DELAY)

    # Atomic decrement if stock >= qty
    if req.atomic_update is None or req.atomic_update is True:

        res = inventory_col.find_one_and_update(
            {"item": req.item, "stock": {"$gte": req.qty}},
            {"$inc": {"stock": -req.qty}},
            return_document=ReturnDocument.AFTER
        )
    else:
        doc = inventory_col.find_one({"item": req.item})

        if doc and doc.get("stock", 0) >= req.qty:

            new_stock = doc["stock"] - req.qty
            inventory_col.update_one(
                {"item": req.item},
                {"$set": {"stock": new_stock}}
            )
            res = inventory_col.find_one({"item": req.item})
        else:
            res = None

    if res:
        remaining = res["stock"]
        return {"reserved": True, "item": req.item, "remaining": remaining}
    else:
        # fetch current stock
        cur = inventory_col.find_one({"item": req.item})
        remaining = cur["stock"] if cur else 0
        return {"reserved": False, "item": req.item, "remaining": remaining}

@app.post("/reset")
def reset(req: ReserveReq):
    """
    Reset inventory for testing: set stock = qty for given item.
    """
    inventory_col.update_one({"item": req.item}, {"$set": {"stock": req.qty}}, upsert=True)
    return {"ok": True, "item": req.item, "stock": req.qty}

@app.get("/health")
def health():
    return {"ok": True}
