from fastapi import FastAPI
from pymongo import MongoClient
import time, uuid

app = FastAPI(title="Inventory Service")
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]


@app.post("/clear_stocks")
def clear_orders():
    db.inventory.delete_many({})


@app.post("/init_stock")
def init_stock(request: dict):
    item: str = request["item"]
    db.inventory.insert_one({"item": item, "stock": 10})


@app.post("/reserve")
def reserve(request: dict):
    item: str = request["item"]
    qty: int = request["qty"]
    delay: int = request.get("delay", 0)
    stock = db.inventory.find_one({"item": item}) or {"item": item, "stock": 10}
    if stock["stock"] >= qty:
        reservation_id = str(uuid.uuid4())
        db.inventory.update_one({"item": item}, {"$inc": {"stock": -qty}}, upsert=True)

        time.sleep(delay)  # Injected delay
        return {"status": "reserved", "reservation_id": reservation_id}
    else:
        time.sleep(delay)  # Injected delay
        return {"status": "out_of_stock", "reservation_id": None}


@app.get("/debug_stock")
def debug_stock(item: str):
    stock = db.inventory.find_one({"item": item}) or {"item": item, "stock": 0}
    return {"item": item, "stock": stock["stock"]}
