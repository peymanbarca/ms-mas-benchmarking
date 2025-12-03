# order_service.py
import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "retail_exp4")
ORDERS_COLL = os.environ.get("ORDERS_COLL", "orders")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
orders_col = db[ORDERS_COLL]

app = FastAPI(title="Order Service")

class CreateReq(BaseModel):
    order_id: str | None = None
    item: str
    qty: int

@app.on_event("startup")
def startup():
    orders_col.create_index("status")
    # orders_col.create_index("_id", unique=True)

@app.post("/create")
def create_order(req: CreateReq):
    order_id = req.order_id or str(uuid.uuid4())
    order_doc = {"_id": order_id, "item": req.item, "qty": req.qty, "status": "INIT", "created_at": time.time()}
    try:
        orders_col.insert_one(order_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"order_id": order_id, "status": "INIT"}

@app.post("/update")
def update_order(body: dict):
    order_id = body.get("order_id")
    status = body.get("status")
    if not order_id or not status:
        raise HTTPException(status_code=400, detail="order_id and status required")
    orders_col.update_one({"_id": order_id}, {"$set": {"status": status}})
    return {"order_id": order_id, "status": status}

@app.post("/reset")
def reset_orders():
    orders_col.delete_many({})
    return {"ok": True}
