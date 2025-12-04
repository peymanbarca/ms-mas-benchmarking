# inventory_microservice.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Literal

# --- Configuration ---
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "supply_chain_db"
SKU_ID = "A101"
app = FastAPI()


# --- Pydantic Schemas ---
class ReservationRequest(BaseModel):
    order_id: str
    quantity: int = Field(gt=0, description="Quantity must be positive.")


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

    # Initialize inventory document if it doesn't exist
    await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID},
        {'$set': {'sku_id': SKU_ID, 'available_stock': 10, 'reserved_stock': 0}},
        upsert=True
    )
    print(f"MongoDB initialized. {SKU_ID} stock set to 10 available.")


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


# --- API Endpoints (The Tool Functions) ---

@app.put("/inventory/reserve")
async def reserve_inventory(request: ReservationRequest):
    """Atomically attempts to move stock from available to reserved."""
    quantity = request.quantity

    # Atomic operation: Filter by sufficient stock AND update
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'available_stock': {'$gte': quantity}},
        {'$inc': {'available_stock': -quantity, 'reserved_stock': quantity}}
    )

    if result.modified_count == 1:
        # Update order status in orders collection (Local transaction for the microservice)
        await app.mongodb["orders"].update_one(
            {'order_id': request.order_id},
            {'$set': {'status': 'RESERVED', 'quantity': quantity, 'sku_id': SKU_ID}},
            upsert=True
        )
        return UpdateResult(success=True, status_code="RESERVED", message="Stock successfully reserved.")
    else:
        # Failure due to stock or race condition (Phantom Consistency prevention)
        return UpdateResult(success=False, status_code="INSUFFICIENT_STOCK",
                            message="Atomic reservation failed. Stock unavailable.")


@app.post("/inventory/confirm")
async def confirm_fulfillment(request: ReservationID):
    """Atomically confirms fulfillment, reducing reserved stock."""
    # Retrieve order quantity to perform the update
    order_doc = await app.mongodb["orders"].find_one({'order_id': request.order_id})
    if not order_doc:
        return UpdateResult(success=False, status_code="ERROR", message="Order not found for confirmation.")
    quantity = order_doc['quantity']

    # Atomic operation: Filter by reserved stock AND update
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity}},
        {'$inc': {'reserved_stock': -quantity}}
    )

    if result.modified_count == 1:
        await app.mongodb["orders"].update_one({'order_id': request.order_id}, {'$set': {'status': 'CONFIRMED'}})
        return UpdateResult(success=True, status_code="ORDER_CONFIRMED", message="Order confirmed. Stock finalized.")
    else:
        # Reserved stock mismatch (Phantom Consistency failure)
        return UpdateResult(success=False, status_code="ERROR",
                            message="Confirmation failed. Reserved stock not found.")


@app.post("/inventory/compensate")
async def compensate_inventory(request: ReservationID):
    """Atomically releases reserved stock back to available stock (Compensation/Rollback)."""
    # Retrieve order quantity
    order_doc = await app.mongodb["orders"].find_one({'order_id': request.order_id})
    if not order_doc or order_doc.get('status') == 'CANCELLED':
        return UpdateResult(success=True, status_code="COMPENSATED", message="Order already cancelled or not found.")

    quantity = order_doc['quantity']

    # Atomic operation: Move stock from reserved back to available
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity}},
        {'$inc': {'available_stock': quantity, 'reserved_stock': -quantity}}
    )

    if result.modified_count == 1:
        await app.mongodb["orders"].update_one({'order_id': request.order_id}, {'$set': {'status': 'COMPENSATED'}})
        return UpdateResult(success=True, status_code="COMPENSATED",
                            message="Stock successfully released (rollback complete).")
    else:
        return UpdateResult(success=False, status_code="ERROR", message="Compensation failed. Reserved stock mismatch.")

# To run: uvicorn inventory_microservice:app --reload --port 8000