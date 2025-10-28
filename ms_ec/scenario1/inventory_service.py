from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import time, uuid, random

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
    drop_rate: int = request.get("drop_rate", 0)

    stock = db.inventory.find_one({"item": item}) or {"item": item, "stock": 10}
    if stock["stock"] >= qty:
        reservation_id = str(uuid.uuid4())
        db.inventory.update_one({"item": item}, {"$inc": {"stock": -qty}}, upsert=True)

        time.sleep(delay)  # Injected delay

        # drop response message after db update
        if drop_rate > 0 and random.randint(0, 99) < drop_rate:
            time.sleep(delay)  # Still include delay before reporting failure
            print(f"FAULT INJECTED: Drop Rate triggered (item={item}, rate={drop_rate}%)")
            # Simulates a dropped connection or corrupted response that results in an immediate error
            raise HTTPException(
                status_code=503,
                detail=f"Simulated Service Drop/Error. Drop Rate: {drop_rate}%"
            )

        return {"status": "reserved", "reservation_id": reservation_id}
    else:
        time.sleep(delay)  # Injected delay
        # drop response message after db update
        if drop_rate > 0 and random.randint(0, 99) < drop_rate:
            time.sleep(delay)  # Still include delay before reporting failure
            print(f"FAULT INJECTED: Drop Rate triggered (item={item}, rate={drop_rate}%)")
            # Simulates a dropped connection or corrupted response that results in an immediate error
            raise HTTPException(
                status_code=503,
                detail=f"Simulated Service Drop/Error. Drop Rate: {drop_rate}%"
            )
        return {"status": "out_of_stock", "reservation_id": None}


@app.get("/debug_stock")
def debug_stock(item: str):
    stock = db.inventory.find_one({"item": item}) or {"item": item, "stock": 0}
    return {"item": item, "stock": stock["stock"]}
