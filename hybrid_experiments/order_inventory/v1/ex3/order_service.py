# order_service.py
import asyncio
import httpx  # Asynchronous HTTP client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Literal, Dict, Any

# --- Configuration ---
MONGO_URL = "mongodb://localhost:27017"
INVENTORY_SERVICE_URL = "http://localhost:8000"
app = FastAPI(title="Order Microservice")


# --- Pydantic Schemas ---
class NewOrderRequest(BaseModel):
    order_id: str
    quantity: int = Field(gt=0)
    sku_id: str = "A101"  # Fixed for this experiment


class UpdateResult(BaseModel):
    success: bool
    status_code: Literal
    message: str


# --- MongoDB Setup (Order Service manages 'orders' collection) ---
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGO_URL)
    app.mongodb = app.mongodb_client
    app.http_client = httpx.AsyncClient()
    print("Order Service: DB and HTTP Client initialized.")


@app.on_event("shutdown")
async def shutdown_db_client():
    await app.http_client.aclose()
    app.mongodb_client.close()


# --- SAGA Coordinator Endpoint ---

async def call_inventory_service(url_path: str, data: Dict[str, Any], method: str = 'POST') -> UpdateResult:
    """Helper to call Inventory Service API, handles connection errors."""
    full_url = INVENTORY_SERVICE_URL + url_path
    try:
        if method == 'PUT':
            response = await app.http_client.put(full_url, json=data)
        elif method == 'POST':
            response = await app.http_client.post(full_url, json=data)

        response.raise_for_status()
        return UpdateResult.parse_obj(response.json())

    except httpx.HTTPStatusError as e:
        # Handles 4xx/5xx errors from the microservice
        print(f"SAGA Failure: Inventory Service returned HTTP error: {e.response.status_code}")
        # Try to parse structured error body if available
        try:
            return UpdateResult.parse_obj(e.response.json())
        except Exception:
            return UpdateResult(success=False, status_code="EXTERNAL_ERROR",
                                message=f"HTTP Error: {e.response.status_code}")
    except Exception as e:
        # Handles network or connection errors (systemic non-determinism)
        return UpdateResult(success=False, status_code="NETWORK_ERROR", message=f"Network or connection error: {e}")


@app.post("/orders/create")
async def create_order(request: NewOrderRequest):
    """Coordinates the two-step distributed transaction (SAGA)."""
    order_id = request.order_id

    # 1. Start SAGA: Local Transaction in Order Service (Log PENDING status)
    await app.mongodb["orders"].update_one(
        {'order_id': order_id},
        {'$set': {'status': 'PENDING', 'sku_id': request.sku_id, 'quantity': request.quantity}},
        upsert=True
    )

    # --- Step 1: Reserve Inventory (Call Inventory Service) ---
    reserve_data = {"order_id": order_id, "quantity": request.quantity}
    reserve_result = await call_inventory_service("/inventory/reserve", reserve_data, method='PUT')

    if not reserve_result.success:
        # Reservation failed (e.g., INSUFFICIENT_STOCK or NETWORK_ERROR)
        await app.mongodb["orders"].update_one({'order_id': order_id}, {'$set': {'status': 'CANCELLED'}})
        return {"order_id": order_id, "status": "FAILED", "reason": reserve_result.message}

    # --- Step 2: Confirm Fulfillment (Simulated Success/Next Step) ---

    # Log: Reservation successful (Local Transaction 2)
    await app.mongodb["orders"].update_one({'order_id': order_id}, {'$set': {'status': 'RESERVED'}})

    # Call Inventory Service to confirm
    confirm_data = {"order_id": order_id}
    confirm_result = await call_inventory_service("/inventory/confirm", confirm_data, method='POST')

    if confirm_result.success:
        # Final success
        await app.mongodb["orders"].update_one({'order_id': order_id}, {'$set': {'status': 'CONFIRMED'}})
        return {"order_id": order_id, "status": "CONFIRMED", "message": "Order successfully processed."}
    else:
        # --- Compensation Required (SAGA Rollback) ---
        print(f"!!! SAGA FAILURE: Confirmation failed. Initiating Compensation.")
        compensate_result = await call_inventory_service("/inventory/compensate", confirm_data, method='POST')

        # Log compensation status
        final_status = 'COMPENSATED' if compensate_result.success else 'ROLLBACK_FAILED'
        await app.mongodb["orders"].update_one({'order_id': order_id}, {'$set': {'status': final_status}})

        return {"order_id": order_id, "status": final_status,
                "reason": "Compensation initiated due to confirmation failure."}

# To run:
# 1. Run Inventory Service: uvicorn inventory_service:app --reload --port 8000
# 2. Run Order Service: uvicorn order_service:app --reload --port 8001