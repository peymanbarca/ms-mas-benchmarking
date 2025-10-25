import json
import subprocess
import sys
import uuid

SERVER_SCRIPT_PATH = "./server_mcp.py"


def get_raw_json_response():
    """
    Executes the server as a subprocess and captures the raw stdout/stderr
    to reveal the exact JSON-RPC response string.
    """
    request_id = str(uuid.uuid4())

    # 1. Construct the exact JSON-RPC 2.0 Request Payload
    request_payload_dict = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": "reserve_stock",
            "argument": {
                "order_id": request_id,
                "item": "Laptop",
                "qty": 2
            }
        }
    }
    request_payload_json = json.dumps(request_payload_dict)

    print(f"--- Sending Raw Request ---\n{request_payload_json}\n")

    # 2. Execute the Server Subprocess
    cmd = [sys.executable, SERVER_SCRIPT_PATH]
    try:
        result = subprocess.run(
            cmd,
            input=request_payload_json,
            capture_output=True,
            encoding='utf-8',  # Decode STDOUT/STDERR
            check=True  # Raise error on non-zero exit code
        )

        # 3. Extract the Raw JSON-RPC Response from STDOUT
        # The response is usually the first non-empty line on stdout
        raw_output_lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]

        if raw_output_lines:
            raw_response_json = raw_output_lines[0]
            print("=" * 50)
            print("✅ **EXACT RAW JSON-RPC RESPONSE**")
            print(raw_response_json)
            print(f"Size: {len(raw_response_json.encode('utf-8'))} bytes")
            print("=" * 50)

            # Optional: Deserialize and inspect the result body
            response_dict = json.loads(raw_response_json)
            print("\nResponse Dictionary (for structure):")
            print(json.dumps(response_dict, indent=2))

            if result.stderr:
                print("\nServer STDERR (Logs/Warnings):")
                print(result.stderr)

            return raw_response_json

        else:
            print("❌ Server returned empty STDOUT.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Subprocess Error (Code: {e.returncode}):")
        print(f"STDERR: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print("❌ Could not decode the raw output as JSON.")
        return None


if __name__ == "__main__":
    get_raw_json_response()