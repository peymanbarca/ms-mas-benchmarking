from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient, ReturnDocument
import time
import threading

app = FastAPI()
inventory_col = MongoClient("mongodb://user:pass1@localhost:27017/")["retail_exp3"]["inventory"]
LOCK = threading.Lock()  # simple lock to avoid race conditions for demo

class Reservation(BaseModel):
    order_id: str
    item: str
    qty: int
    atomic_update: bool = False
    delay: float = 0.0
    drop: int = 0

@app.post("/reset_stocks")
def reset_stocks(request: dict):
    inventory_col.delete_many({})
    item: str = request["item"]
    stock: int = request["stock"]
    inventory_col.insert_one({"item": item, "stock": stock})



@app.post("/reserve")
def reserve_stock(req: Reservation):
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
        remaining = res["stock"]
        return {"order_id": req.order_id, "status": "reserved", "remaining": remaining}
    else:
        # fetch current stock
        cur = inventory_col.find_one({"item": req.item})
        remaining = cur["stock"] if cur else 0
        return {"order_id": req.order_id, "status": "out_of_stock", "remaining": remaining}

    # with LOCK:
    #     stock_doc = INVENTORY_DB.find_one({"item": reservation.item})
    #     if not stock_doc:
    #         # initialize stock
    #         INVENTORY_DB.insert_one({"item": reservation.item, "stock": 10})
    #         stock = 10
    #     else:
    #         stock = stock_doc["stock"]
    #     if stock >= reservation.qty:
    #         new_stock = stock - reservation.qty
    #         INVENTORY_DB.update_one({"item": reservation.item}, {"$set": {"stock": new_stock}})
    #         return {"order_id": reservation.order_id, "status": "reserved"}
    #     else:
    #         return {"order_id": reservation.order_id, "status": "out_of_stock"}
