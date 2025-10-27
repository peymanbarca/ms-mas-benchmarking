import asyncio
import time
import httpx  # Requires: pip install httpx
import uuid
from typing import List, Dict, Any

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/order"
NUM_TRIALS = 100


async def run_single_call(client: httpx.AsyncClient, trial_index: int) -> Dict[str, Any]:
    """
    Makes a single asynchronous POST request to the Order Service API.
    """
    order_data = {
        "item": f"Laptop",
        "qty": 2
    }

    start_request = time.perf_counter()
    try:
        # The AsyncClient allows this call to be non-blocking
        response = await client.post(API_URL, json=order_data)
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes
        end_request = time.perf_counter()
        print(response, (end_request - start_request) * 1000)


        return {
            "trial": trial_index,
            "success": True,
            "status": response.json().get("status", "N/A"),
            "latency_ms": (end_request - start_request) * 1000,
            "response_status": response.status_code
        }
    except httpx.HTTPStatusError as e:
        end_request = time.perf_counter()
        return {
            "trial": trial_index,
            "success": False,
            "error": f"HTTP Error: {e.response.status_code}",
            "latency_ms": (end_request - start_request) * 1000,
            "response_status": e.response.status_code
        }
    except Exception as e:
        end_request = time.perf_counter()
        return {
            "trial": trial_index,
            "success": False,
            "error": str(e),
            "latency_ms": (end_request - start_request) * 1000,
            "response_status": 0
        }


async def main():
    """
    Runs NUM_TRIALS concurrent calls and reports performance metrics.
    """
    print(f"--- Starting {NUM_TRIALS} Concurrent Trials against {API_URL} ---")

    start_time = time.perf_counter()

    # Use httpx.AsyncClient for connection pooling and concurrency
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create a list of 100 coroutines
        tasks = [run_single_call(client, i) for i in range(1, NUM_TRIALS + 1)]

        # Run all tasks concurrently
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    # --- Process Summary ---
    success_count = sum(1 for res in results if res['success'])
    failure_count = NUM_TRIALS - success_count

    # Calculate average latency from successful requests
    successful_latencies = [res['latency_ms'] for res in results if res['success']]
    average_latency_ms = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0

    print("\n" + "=" * 60)
    print("📊 **EXTERNAL CONCURRENCY TEST RESULTS**")
    print(f"API Target: {API_URL}")
    print(f"Trials Completed: {NUM_TRIALS}")
    print(f"Network Successes: {success_count}, Failures: {failure_count}")
    print(f"Total Response Time (for all trials): {total_time_ms:.2f} ms")
    print(f"Average Latency (per successful call): {average_latency_ms:.2f} ms")
    print("=" * 60)

    # if failure_count > 0:
    #     print(f"\nSample Errors (First {min(failure_count, 5)}):")
    #     for res in results:
    #         if not res['success']:
    #             print(f"  Trial {res['trial']}: {res.get('error', 'Unknown Error')} (Status: {res['response_status']})")
    #             if failure_count <= 5: break  # Limit output


if __name__ == "__main__":
    asyncio.run(main())
