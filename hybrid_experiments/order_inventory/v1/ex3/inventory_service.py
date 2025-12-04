# inventory_service.py
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

    # Initialize inventory stock
    await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID},
        {'$set': {'sku_id': SKU_ID, 'available_stock': 10, 'reserved_stock': 0}},
        upsert=True
    )
    print(f"Inventory Service: DB initialized. {SKU_ID} stock set to 10 available.")


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


# --- API Endpoints (Local Transactions) ---

@app.put("/inventory/reserve")
async def reserve_inventory(request: ReservationRequest):
    """Local Transaction 1: Atomically moves stock from available to reserved."""
    quantity = request.quantity

    # Atomic operation: Filter by sufficient stock AND update
    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'available_stock': {'$gte': quantity}},
        {'$inc': {'available_stock': -quantity, 'reserved_stock': quantity}}
    )

    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="RESERVED", message="Stock reserved.")
    else:
        return UpdateResult(success=False, status_code="INSUFFICIENT_STOCK", message="Atomic reservation failed.")


@app.post("/inventory/confirm")
async def confirm_fulfillment(request: ReservationID):
    """Local Transaction 2: Confirms fulfillment (reducing reserved stock)."""
    # Note: Quantity would typically be looked up from an Order document here.
    # We assume a successful prior reservation means we have the quantity.
    # For simplicity, we'll confirm 3 units reserved from the order service call.
    quantity_to_confirm = 3

    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity_to_confirm}},
        {'$inc': {'reserved_stock': -quantity_to_confirm}}
    )

    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="ORDER_CONFIRMED", message="Order confirmed. Stock finalized.")
    else:
        return UpdateResult(success=False, status_code="ERROR", message="Confirmation failed. Reserved stock mismatch.")


@app.post("/inventory/compensate")
async def compensate_inventory(request: ReservationID):
    """Compensating Transaction: Releases reserved stock back to available."""
    # We assume the Order Service tells us how much to release (e.g., 3 units)
    quantity_to_release = 3

    result = await app.mongodb["inventory"].update_one(
        {'sku_id': SKU_ID, 'reserved_stock': {'$gte': quantity_to_release}},
        {'$inc': {'available_stock': quantity_to_release, 'reserved_stock': -quantity_to_release}}
    )

    if result.modified_count == 1:
        return UpdateResult(success=True, status_code="COMPENSATED", message="Stock released (rollback complete).")
    else:
        return UpdateResult(success=False, status_code="ERROR", message="Compensation failed.")