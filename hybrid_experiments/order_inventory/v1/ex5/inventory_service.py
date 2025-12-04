# inventory_service.py (Run this first on port 8000)
from fastapi import FastAPI
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Literal

# --- Configuration ---
MONGO_URL = "mongodb://localhost:27017"
SKU_ID = "A101"
app = FastAPI(title="Inventory Microservice")

# --- Schemas ---
class ReservationRequest(BaseModel):
    order_id: str
    quantity: int = Field(gt=0)
class ReservationID(BaseModel):
    order_id: str
class UpdateResult(BaseModel):
    success: bool
    status_code: Literal
    message: str

# --- MongoDB Setup ---
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGO_URL)
    app.mongodb = app.mongodb_client
    await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID},
        {'$set': {'sku_id': SKU_ID, 'available_stock': 10, 'reserved_stock': 0}},
        upsert=True
    )

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

# --- API Endpoints ---
@app.put("/inventory/reserve")
async def reserve_inventory(request: ReservationRequest):
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'available_stock': {'$gte': request.quantity}},
        {'$inc': {'available_stock': -request.quantity, 'reserved_stock': request.quantity}}
    )
    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="RESERVED", message="Stock successfully reserved.")
    else:
        return UpdateResult(success=False, status_code="INSUFFICIENT_STOCK", message="Atomic reservation failed.")

@app.post("/inventory/confirm_deduct")
async def confirm_fulfillment(request: ReservationID):
    # This assumes the Order Service already verified the quantity
    # Simple reduction of reserved stock
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gt': 0}},
        {'$inc': {'reserved_stock': -request.quantity}}
    )
    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="DEDUCTED", message="Reserved stock deducted.")
    return UpdateResult(success=False, status_code="ERROR", message="Deduction failed.")

@app.post("/inventory/compensate")
async def compensate_inventory(request: ReservationID):
    # Simple rollback, moving reserved stock back to available
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gt': 0}},
        {'$inc': {'available_stock': request.quantity, 'reserved_stock': -request.quantity}}
    )
    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="COMPENSATED", message="Stock released (rollback complete).")
    return UpdateResult(success=False, status_code="ERROR", message="Compensation failed.")