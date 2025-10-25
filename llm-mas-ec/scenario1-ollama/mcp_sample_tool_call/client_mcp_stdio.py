import asyncio
import json
import time
import uuid
import sys

from fastmcp import Client

SERVER_SCRIPT_PATH = "./server_mcp.py"
TOOL_NAME = "reserve_stock"
NUM_TRIALS = 100


def get_tool_call_payload(item_name: str, quantity: int, request_id: str) -> str:
    """
    Constructs the exact raw JSON-RPC 2.0 payload to be sent.
    """
    # The structure must adhere to JSON-RPC 2.0 for tools/call
    request_dict = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": TOOL_NAME,
            "argument": {
                "order_id": request_id,
                "item": item_name,
                "qty": quantity
            }
        }
    }

    # Serialize to JSON string
    request_payload_json = json.dumps(request_dict)

    return request_payload_json


async def run_client_call(item_name: str, quantity: int):
    """
    Connects to the local server script via the FastMCP Client,
    which automatically uses StdioTransport.
    """
    order_id = str(uuid.uuid4())

    print(f"--- Calling Tool '{TOOL_NAME}' for Order: {order_id} ---")

    # 1. Create the Client instance, passing the server script path.
    # The Client automatically infers the StdioTransport and the 'python' command.
    # It will launch the mcp_server.py script as a subprocess.
    client = Client(SERVER_SCRIPT_PATH)

    # 2. Use the async context manager to manage the connection lifecycle.
    # This block handles starting the subprocess and ensuring it's cleaned up.
    start_time = time.perf_counter()
    try:
        async with client:
            print(f"Attempting to connect to server at: {SERVER_SCRIPT_PATH}")

            # 3. Call the tool using the high-level 'call_tool' method
            # Arguments are passed as a dictionary.
            result = await client.call_tool(
                name=TOOL_NAME,
                arguments={
                    "order_id": order_id,
                    "item": item_name,
                    "qty": quantity
                }
            )

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # The result object contains structured data and response metadata
            print(f"✅ Tool call successful in {response_time_ms:.2f} ms.")

            # Use result to get the actual Python dictionary returned by the tool function
            print("\nTool Output:")
            print(json.dumps(json.loads(result.content[0].text), indent=2))

    except Exception as e:
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        print(f"❌ An error occurred during the tool call in {response_time_ms:.2f} ms:")
        print(f"Error: {e}")


async def run_single_call(trial_index: int, item_name: str, quantity: int, client: Client):
    """Performs a single, measured client call."""

    request_id = str(uuid.uuid4())

    # 1. Get and Log the Payload (only for the first trial)
    if trial_index == 1:
        # Generate the exact JSON payload the client will send
        payload_json = get_tool_call_payload(item_name, quantity, request_id)
        payload_size_bytes = len(payload_json.encode('utf-8'))

        print("-" * 50)
        print(f"📦 **Trial 1 Payload (JSON-RPC 2.0 Sent)**:")
        print(payload_json)
        print(f"📦 **Payload Size**: {payload_size_bytes} bytes")

    # 2. Perform the actual call using the high-level client function
    try:
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

        if trial_index == 1:
            res_payload_size_bytes = len(json.dumps(final_response).encode('utf-8'))
            print(f"📦 **Payload Size**: {res_payload_size_bytes} bytes")
            print(f"📦 **Total Payload Size**: {res_payload_size_bytes + payload_size_bytes} bytes")

    except Exception as e:
        return f"Trial {trial_index} ERROR (Protocol): {e}"


async def run_concurrent_test():
    """Manages the concurrent execution of 100 client calls."""

    # Create a single client instance that the tasks will share
    # Using the path automatically selects StdioTransport and manages the subprocess.
    client = Client(SERVER_SCRIPT_PATH)

    # We use the context manager to ensure the server subprocess starts
    # before we start sending tasks and is properly shut down afterwards.
    print(f"Starting {NUM_TRIALS} concurrent trials...")
    start_time = time.perf_counter()

    try:
        async with client:
            # Create a list of 100 coroutines (tasks)
            tasks = [
                run_single_call(
                    trial_index=i,
                    item_name="Laptop",
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
            print("📊 **CONCURRENCY TEST RESULTS**")
            print(f"Trials Completed: {NUM_TRIALS}")
            print(f"Total Time: {total_time_ms:.2f} ms")
            print(f"Average Latency: {(total_time_ms / NUM_TRIALS):.2f} ms per call")
            print("=" * 50)

            # Print a sample of results
            # print("\nSample results (first 5):", results[:5])

    except FileNotFoundError:
        print(f"\n❌ Error: Server script '{SERVER_SCRIPT_PATH}' not found. Ensure both files are saved.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Reservation 1 case
    # asyncio.run(run_client_call(item_name="Laptop", quantity=5))

    print("\n" + "=" * 50 + "\n")

    asyncio.run(run_concurrent_test())
