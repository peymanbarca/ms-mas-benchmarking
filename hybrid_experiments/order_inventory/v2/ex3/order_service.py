from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from pymongo import MongoClient
import uuid
import time

app = FastAPI()
ORDER_DB = MongoClient("mongodb://user:pass1@localhost:27017/")["retail_exp3"]["orders"]
INVENTORY_SERVICE_URL = "http://127.0.0.1:8001/reserve"  # inventory service endpoint


class Order(BaseModel):
    item: str
    qty: int


@app.post("/clear_orders")
def clear_orders():
    ORDER_DB.delete_many({})


@app.post("/create_order")
def create_order(order: Order):
    order_id = str(uuid.uuid4())
    start_time = time.time()
    # Save order INIT
    ORDER_DB.insert_one({"_id": order_id, "item": order.item, "qty": order.qty, "status": "INIT"})

    # Call inventory service
    try:
        resp = requests.post(INVENTORY_SERVICE_URL, json={"order_id": order_id, "item": order.item, "qty": order.qty})
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Inventory service error")
        result = resp.json()
        # Update order status
        ORDER_DB.update_one({"_id": order_id}, {"$set": {"status": result["status"]}})
    except Exception as e:
        ORDER_DB.update_one({"_id": order_id}, {"$set": {"status": "error"}})
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    latency = end_time - start_time
    return {"order_id": order_id, "status": result["status"], "latency": latency}
