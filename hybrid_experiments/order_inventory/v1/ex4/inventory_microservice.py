# inventory_microservice.py (Run this file first on port 8000)
from fastapi import FastAPI
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Literal

# --- Configuration ---
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "supply_chain_db"
SKU_ID = "A101"
app = FastAPI(title="Inventory Microservice")


# --- Pydantic Schemas ---
class ReservationRequest(BaseModel):
    order_id: str
    quantity: int = Field(gt=0)
    action: Literal  # Added for flexibility in the tool wrapper


class ReservationID(BaseModel):
    order_id: str
    action: Literal


class UpdateResult(BaseModel):
    success: bool
    status_code: Literal
    message: str


# --- MongoDB Setup ---
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGO_URL)
    app.mongodb = app.mongodb_client
    # Initialize inventory document
    await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID},
        {'$set': {'sku_id': SKU_ID, 'available_stock': 10, 'reserved_stock': 0}},
        upsert=True
    )
    print(f"Inventory Service: DB initialized. {SKU_ID} stock set to 10 available.")


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


# --- API Endpoints ---

@app.put("/inventory/reserve")
async def reserve_inventory(request: ReservationRequest):
    """Atomically attempts to move stock from available to reserved."""
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'available_stock': {'$gte': request.quantity}},
        {'$inc': {'available_stock': -request.quantity, 'reserved_stock': request.quantity}}
    )
    if result.modified_count == 1:
        await app.mongodb["orders"].update_one(
            {'order_id': request.order_id},
            {'$set': {'status': 'RESERVED', 'quantity': request.quantity, 'sku_id': SKU_ID}},
            upsert=True
        )
        return UpdateResult(success=True, status_code="RESERVED", message="Stock successfully reserved.")
    else:
        return UpdateResult(success=False, status_code="INSUFFICIENT_STOCK",
                            message="Atomic reservation failed. Stock unavailable.")


@app.post("/inventory/confirm")
async def confirm_fulfillment(request: ReservationID):
    """Atomically confirms fulfillment, reducing reserved stock."""
    order_doc = await app.mongodb["orders"].find_one({'order_id': request.order_id})
    if not order_doc:
        return UpdateResult(success=False, status_code="ERROR", message="Order not found for confirmation.")
    quantity = order_doc['quantity']

    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity}},
        {'$inc': {'reserved_stock': -quantity}}
    )

    if result.modified_count == 1:
        await app.mongodb["orders"].update_one({'order_id': request.order_id}, {'$set': {'status': 'CONFIRMED'}})
        return UpdateResult(success=True, status_code="ORDER_CONFIRMED", message="Order confirmed. Stock finalized.")
    else:
        return UpdateResult(success=False, status_code="ERROR",
                            message="Confirmation failed. Reserved stock not found.")


@app.post("/inventory/compensate")
async def compensate_inventory(request: ReservationID):
    """Compensation/Rollback logic (simplified)."""
    order_doc = await app.mongodb["orders"].find_one({'order_id': request.order_id})
    if not order_doc:
        return UpdateResult(success=False, status_code="ERROR", message="Order not found for compensation.")

    quantity = order_doc['quantity']

    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity}},
        {'$inc': {'available_stock': quantity, 'reserved_stock': -quantity}}
    )

    if result.modified_count == 1:
        await app.mongodb["orders"].update_one({'order_id': request.order_id}, {'$set': {'status': 'COMPENSATED'}})
        return UpdateResult(success=True, status_code="COMPENSATED", message="Stock released (rollback complete).")
    else:
        return UpdateResult(success=False, status_code="ERROR", message="Compensation failed.")