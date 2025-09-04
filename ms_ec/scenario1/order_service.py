from fastapi import FastAPI
from pymongo import MongoClient
import requests, uuid

app = FastAPI(title="Order Service")
client = MongoClient("mongodb://user:pass1@localhost:27017/")
db = client["retail"]


@app.post("/clear_orders")
def clear_orders():
    db.orders.delete_many({})


@app.post("/order")
def create_order(request: dict):
    item: str = request["item"]
    qty: int = request["qty"]
    order_id = str(uuid.uuid4())
    db.orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT"})

    # Call Inventory Service
    r = requests.post("http://localhost:8082/reserve", json={"item": item, "qty": qty})
    res = r.json()

    if res["status"] == "reserved":
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "RESERVED"}})
        # Mock payment success
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "COMPLETED"}})
    elif res["status"] == "out_of_stock":
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED_OUT_OF_STOCK"}})
    else:
        db.orders.update_one({"_id": order_id}, {"$set": {"status": "FAILED"}})

    return {"order_id": order_id, "final_status": db.orders.find_one({"_id": order_id})["status"]}
