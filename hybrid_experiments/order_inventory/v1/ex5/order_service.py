# order_service.py (Run this second on port 8001)
from fastapi import FastAPI
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Literal

# --- Configuration ---
MONGO_URL = "mongodb://localhost:27017"
app = FastAPI(title="Order Microservice")

# --- Schemas ---
class OrderRequest(BaseModel):
    order_id: str
    quantity: int = Field(gt=0)
    sku_id: str = "A101"

class StatusUpdate(BaseModel):
    order_id: str
    status: Literal

@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGO_URL)
    app.mongodb = app.mongodb_client

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

# --- API Endpoints ---

@app.post("/orders/create")
async def create_order(request: OrderRequest):
    """Creates the order record with PENDING status."""
    await app.mongodb["orders"].update_one(
        {'order_id': request.order_id},
        {'$set': {'status': 'PENDING', 'sku_id': request.sku_id, 'quantity': request.quantity}},
        upsert=True
    )
    return {"order_id": request.order_id, "status": "PENDING"}

@app.post("/orders/update_status")
async def update_order_status(request: StatusUpdate):
    """Updates the order status, providing a shared state checkpoint."""
    result = await app.mongodb["orders"].update_one(
        {'order_id': request.order_id},
        {'$set': {'status': request.status}}
    )
    return {"order_id": request.order_id, "status": request.status, "modified": result.modified_count}

@app.get("/orders/{order_id}")
async def get_order_details(order_id: str):
    """Retrieves order details (used by Inventory Agent for lookup)."""
    order_doc = await app.mongodb["orders"].find_one({'order_id': order_id}, {'_id': 0})
    if order_doc:
        return order_doc
    raise HTTPException(status_code=404, detail="Order not found")