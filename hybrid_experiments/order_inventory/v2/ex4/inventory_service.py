# inventory_service.py
import os
import time
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient, ReturnDocument

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "retail_exp4")
INVENTORY_COLL = os.environ.get("INVENTORY_COLL", "inventory")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
inventory_col = db[INVENTORY_COLL]

app = FastAPI(title="Inventory Service")

class CheckReq(BaseModel):
    item: str
    qty: int

class ReserveReq(BaseModel):
    order_id: str
    item: str
    qty: int
    atomic_update: bool = False
    delay: float = 0.0
    drop: int = 0

@app.on_event("startup")
def startup():
    inventory_col.create_index("item", unique=True)

@app.post("/check")
def check_stock(req: CheckReq):
    doc = inventory_col.find_one({"item": req.item})
    available = doc["stock"] if doc else 0
    return {"item": req.item, "requested": req.qty, "available": available, "ok": available >= req.qty}

@app.post("/reserve")
def reserve_stock(req: ReserveReq):
    # optional drop injection: simulate network failure by returning 500 occasionally
    INJECT_DELAY, INJECT_DROP_RATE = 0, 0
    if req.delay and req.delay > 0:
        INJECT_DELAY = req.delay
    if req.drop and req.drop > 0:
        INJECT_DROP_RATE = req.drop

    # inject delay
    if INJECT_DELAY > 0:
        time.sleep(INJECT_DELAY)

    if INJECT_DROP_RATE > 0:
        import random
        if random.randint(0, 99) < INJECT_DROP_RATE:
            # simulate dropped request
            raise HTTPException(status_code=503, detail="simulated service drop")

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
        return {"order_id": req.order_id, "status": "reserved", "remaining": res["stock"]}
    else:
        doc = inventory_col.find_one({"item": req.item})
        remaining = doc["stock"] if doc else 0
        return {"order_id": req.order_id, "status": "out_of_stock", "remaining": remaining}

@app.post("/reset")
def reset_stock(body: dict):
    item = body.get("item")
    qty = body.get("qty", 0)
    if item is None:
        raise HTTPException(status_code=400, detail="item required")
    inventory_col.update_one({"item": item}, {"$set": {"stock": qty}}, upsert=True)
    return {"item": item, "stock": qty}
