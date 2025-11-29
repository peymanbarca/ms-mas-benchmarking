from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import threading

app = FastAPI()
INVENTORY_DB = MongoClient("mongodb://user:pass1@localhost:27017/")["retail_exp3"]["inventory"]
# LOCK = threading.Lock()  # simple lock to avoid race conditions for demo

class Reservation(BaseModel):
    order_id: str
    item: str
    qty: int

@app.post("/reset_stocks")
def reset_stocks(request: dict):
    INVENTORY_DB.delete_many({})
    item: str = request["item"]
    stock: int = request["stock"]
    INVENTORY_DB.insert_one({"item": item, "stock": stock})



@app.post("/reserve")
def reserve_stock(reservation: Reservation):
    # with LOCK:
        stock_doc = INVENTORY_DB.find_one({"item": reservation.item})
        if not stock_doc:
            # initialize stock
            INVENTORY_DB.insert_one({"item": reservation.item, "stock": 10})
            stock = 10
        else:
            stock = stock_doc["stock"]
        if stock >= reservation.qty:
            new_stock = stock - reservation.qty
            INVENTORY_DB.update_one({"item": reservation.item}, {"$set": {"stock": new_stock}})
            return {"order_id": reservation.order_id, "status": "reserved"}
        else:
            return {"order_id": reservation.order_id, "status": "out_of_stock"}
