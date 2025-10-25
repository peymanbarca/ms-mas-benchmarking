import asyncio
import json
import time
import uuid

from fastmcp import Client

# Client connects to the URL where the server is running.
SERVER_URL = "http://127.0.0.1:8000/mcp"
TOOL_NAME = "reserve_stock"
NUM_TRIALS = 100  # Can still run concurrency easily with async


async def run_single_call(trial_index: int, item_name: str, quantity: int, client: Client):
    """Performs a single, measured client call against the HTTP server."""

    request_id = str(uuid.uuid4())

    try:
        # call_tool handles the HTTP POST request to the server
        result = await client.call_tool(
            name=TOOL_NAME,
            arguments={
                "order_id": request_id,
                "item": item_name,
                "qty": quantity
            }
        )

        print("\nTool Output:")
        response = json.loads(result.content[0].text)
        final_response = {
              "jsonrpc": "2.0",
              "id": response["order_id"],
              "data": response
        }
        print(json.dumps(final_response))

    except Exception as e:
        return f"Trial {trial_index} ERROR (Network/Protocol): {e}"


async def run_concurrent_test_sse():
    """Manages the concurrent execution of 100 client calls over HTTP."""

    # Crucial change: Initialize client with the server URL.
    # Client will automatically use HttpTransport (which supports SSE).
    client = Client(SERVER_URL)

    print(f"Starting {NUM_TRIALS} concurrent HTTP/SSE trials against {SERVER_URL}...")
    start_time = time.perf_counter()

    try:
        # The context manager now manages the network session lifecycle.
        async with client:
            tasks = [
                run_single_call(
                    trial_index=i,
                    item_name=f"Laptop",
                    quantity=2,
                    client=client
                )
                for i in range(1, NUM_TRIALS + 1)
            ]

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000


            print("\n" + "=" * 50)
            print("📊 **HTTP/SSE CONCURRENCY TEST RESULTS**")
            print(f"Trials Completed: {NUM_TRIALS}")
            print(f"Total Time: {total_time_ms:.2f} ms")
            print(f"Average Latency: {(total_time_ms / NUM_TRIALS):.2f} ms per call")
            print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_concurrent_test_sse())