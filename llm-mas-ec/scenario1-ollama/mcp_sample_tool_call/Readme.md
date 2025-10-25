
MCP server exposing the reservation logic, and a client (from your order agent) invoking it

supporting both stdio transport and WebSocket transport

## MCP Server Implementation

This server exposes a tool reserve_stock(item, qty, order_id) which does the inventory logic + DB updates.


## MCP Client Invocation from Order Agent

In your order_agent (or wherever you previously used DBAgent), replace DB-calls with MCP tool calls.

client.call(tool_name, **args) sends a JSON-RPC request under the hood.

After the call we can inspect client.last_request_bytes, client.last_response_bytes (assuming SDK supports these) for payload size.

## WebSocket vs stdio Transport Differences
Transport	Usage	Notes
stdio	    Local subprocess client/server	Very low latency, no network overhead
websocket	Server listens on WS port, clients connect	Slightly higher latency (network)