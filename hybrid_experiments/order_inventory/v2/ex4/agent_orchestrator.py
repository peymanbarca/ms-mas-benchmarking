
import os
import time
import json
import re
import logging
import random
from typing import Dict, Any

import httpx
import redis

# langgraph StateGraph import (1.0.2 style)
try:
    from langgraph.graph import StateGraph, START
    langgraph_available = True
except Exception:
    langgraph_available = False

# optional Ollama or other LLM wrapper
try:
    from langchain_community.llms import Ollama  # type: ignore
    ollama_available = True
except Exception:
    Ollama = None
    ollama_available = False

logging.basicConfig(level=logging.INFO, filename="exp4_agent_langgraph_v2.log",
                    format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------
# Config
# ----------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASS = os.environ.get("REDIS_PASS", "0")

INVENTORY_CHECK_URL = os.environ.get("INVENTORY_CHECK_URL", "http://localhost:8002/check")
INVENTORY_RESERVE_URL = os.environ.get("INVENTORY_RESERVE_URL", "http://localhost:8002/reserve")
ORDER_CREATE_URL = os.environ.get("ORDER_CREATE_URL", "http://localhost:8001/create")
ORDER_UPDATE_URL = os.environ.get("ORDER_UPDATE_URL", "http://localhost:8001/update_status")

# Fault injection
AGENT_DELAY = float(os.environ.get("AGENT_DELAY", "0.0"))
AGENT_DROP = int(os.environ.get("AGENT_DROP", "0"))  # percent

TOOL_TIMEOUT = float(os.environ.get("TOOL_TIMEOUT", "6.0"))
MAX_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "8"))

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2")

# ----------------------------
# Redis state store
# ----------------------------
class RedisStateStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASS):
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
    def save_state(self, key: str, state: dict, ex: int = 600):
        try:
            self.r.set(key, json.dumps(state), ex=ex)
        except Exception as e:
            logging.error("Redis save error: %s", e)
    def load_state(self, key: str) -> dict:
        try:
            v = self.r.get(key)
            return json.loads(v) if v else {}
        except Exception as e:
            logging.error("Redis load error: %s", e)
            return {}
state_store = RedisStateStore()

# ----------------------------
# LLM client / fallback
# ----------------------------
_llm = None
if ollama_available:
    try:
        _llm = Ollama(model=OLLAMA_MODEL)
    except Exception as e:
        logging.warning("Ollama init failed: %s", e)
        _llm = None

def call_llm_text(prompt: str) -> str:
    """
    Return raw LLM text. If no LLM, deterministic fallback returns a JSON plan for a 'place order' intent.
    The agent uses structured JSON in/out with the LLM.
    """
    if _llm:
        try:
            return str(_llm.invoke(prompt))
        except Exception as e:
            logging.warning("LLM call failed: %s", e)
    # deterministic fallback: simple reactive plan
    # If prompt contains "place order", return initial action check_inventory
    if "place" in prompt.lower() and "order" in prompt.lower():
        return json.dumps({"action": "check_inventory", "args": {}})
    # if prompt contains a tool result that says available, request reserve then create
    if '"available": true' in prompt or '"ok": true' in prompt:
        return json.dumps({"action": "reserve_inventory", "args": {}})
    # default done
    return json.dumps({"action": "done", "result": {"reason": "no_action"}})

def parse_llm_json(text: str) -> dict:
    """Extract first JSON object found in text; otherwise return {}"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(text)
    except Exception:
        logging.warning("Failed to parse LLM output as JSON. Output snippet: %.120s", text)
        return {}

# ----------------------------
# Tool wrappers
# ----------------------------
def _maybe_agent_delay_and_drop():
    if AGENT_DELAY > 0:
        time.sleep(AGENT_DELAY)
    if AGENT_DROP > 0 and random.randint(0, 99) < AGENT_DROP:
        return True
    return False

def tool_check_inventory(item: str, qty: int) -> dict:
    if _maybe_agent_delay_and_drop():
        return {"ok": False, "error": "agent_dropped"}
    try:
        with httpx.Client(timeout=TOOL_TIMEOUT) as c:
            r = c.post(INVENTORY_CHECK_URL, json={"item": item, "qty": qty})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error("tool_check_inventory error: %s", e)
        return {"ok": False, "error": str(e)}

def tool_reserve_inventory(order_id: str, item: str, qty: int) -> dict:
    if _maybe_agent_delay_and_drop():
        return {"status": "error", "error": "agent_dropped"}
    try:
        with httpx.Client(timeout=TOOL_TIMEOUT) as c:
            r = c.post(INVENTORY_RESERVE_URL, json={"order_id": order_id, "item": item, "qty": qty})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error("tool_reserve_inventory error: %s", e)
        return {"status": "error", "error": str(e)}

def tool_create_order(order_id: str, item: str, qty: int) -> dict:
    if _maybe_agent_delay_and_drop():
        return {"status": "error", "error": "agent_dropped"}
    try:
        with httpx.Client(timeout=TOOL_TIMEOUT) as c:
            r = c.post(ORDER_CREATE_URL, json={"order_id": order_id, "item": item, "qty": qty})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error("tool_create_order error: %s", e)
        return {"status": "error", "error": str(e)}

def tool_update_order(order_id: str, status: str) -> dict:
    try:
        with httpx.Client(timeout=TOOL_TIMEOUT) as c:
            r = c.post(ORDER_UPDATE_URL, json={"order_id": order_id, "status": status})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error("tool_update_order error: %s", e)
        return {"status": "error", "error": str(e)}

# ----------------------------
# Orchestrator logic as a Python node (returns partial dicts)
# The node runs an iterative loop:
#   - build prompt from state + history
#   - call LLM -> expect {"action": "...", "args": {...}}
#   - dispatch to tool -> get tool_result
#   - append tool_result to history, save partial state
#   - if action == "done" or steps > MAX_STEPS -> finish
# ----------------------------
def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state (partial) can contain:
      - order_id (optional), generated_order_id
      - item, qty
      - llm_history: list of messages (system/user/tool outputs)
      - trace: list of executed tool steps with latencies/outputs
    Returns partial dict to merge into persistent state.
    """
    # ensure necessary fields
    order_id = state.get("order_id") or state.get("generated_order_id") or (str(int(time.time()*1000)) + "-" + str(os.getpid()))
    item = state.get("item")
    qty = state.get("qty")
    llm_history = state.get("llm_history", [])
    trace = state.get("trace", [])

    # initial user message if first time
    if not llm_history:
        user_msg = f"Place an order for item={item} qty={qty}. Respond with JSON: {{\"action\":\"<tool|done>\", \"args\":{{...}}}}"
        llm_history.append({"role": "user", "content": user_msg})

    steps = 0
    while steps < MAX_STEPS:
        steps += 1
        # build prompt by concatenating history (system then messages)
        prompt_lines = []
        prompt_lines.append("System: You are an orchestration agent. Decide the next action as JSON.")
        for m in llm_history:
            prompt_lines.append(f"{m['role'].upper()}: {m['content']}")
        prompt = "\n\n".join(prompt_lines)

        # call LLM
        llm_text = call_llm_text(prompt)
        parsed = parse_llm_json(llm_text)
        action = parsed.get("action")
        args = parsed.get("args", {})

        # persist LLM decision in history
        llm_history.append({"role": "assistant", "content": json.dumps(parsed)})

        if not action:
            # nothing actionable -> finish
            break

        action = action.lower()

        # dispatch
        if action == "check_inventory":
            t0 = time.time()
            tool_out = tool_check_inventory(item, qty)
            t1 = time.time()
            step_record = {"step": "check_inventory", "out": tool_out, "latency": round(t1 - t0, 4)}
            trace.append(step_record)
            # feed tool result back to LLM (tool role)
            llm_history.append({"role": "tool", "name": "check_inventory", "content": json.dumps(tool_out)})
            # save partial state
            partial = {"generated_order_id": order_id, "order_id": order_id, "item": item, "qty": qty, "llm_history": llm_history, "trace": trace}
            state_store.save_state(f"order:{order_id}", partial)
            # let loop continue (LLM will decide next action based on tool result)

        elif action == "reserve_inventory":
            t0 = time.time()
            tool_out = tool_reserve_inventory(order_id, item, qty)
            t1 = time.time()
            step_record = {"step": "reserve_inventory", "out": tool_out, "latency": round(t1 - t0, 4)}
            trace.append(step_record)
            llm_history.append({"role": "tool", "name": "reserve_inventory", "content": json.dumps(tool_out)})
            partial = {"generated_order_id": order_id, "order_id": order_id, "llm_history": llm_history, "trace": trace}
            state_store.save_state(f"order:{order_id}", partial)

        elif action == "create_order":
            t0 = time.time()
            tool_out = tool_create_order(order_id, item, qty)
            t1 = time.time()
            step_record = {"step": "create_order", "out": tool_out, "latency": round(t1 - t0, 4)}
            trace.append(step_record)
            llm_history.append({"role": "tool", "name": "create_order", "content": json.dumps(tool_out)})
            partial = {"order_id": order_id, "generated_order_id": order_id, "llm_history": llm_history, "trace": trace}
            state_store.save_state(f"order:{order_id}", partial)

        elif action == "update_order":
            status = args.get("status") or "reserved"
            t0 = time.time()
            tool_out = tool_update_order(order_id, status)
            t1 = time.time()
            step_record = {"step": "update_order", "out": tool_out, "latency": round(t1 - t0, 4)}
            trace.append(step_record)
            llm_history.append({"role": "tool", "name": "update_order", "content": json.dumps(tool_out)})
            partial = {"order_id": order_id, "llm_history": llm_history, "trace": trace}
            state_store.save_state(f"order:{order_id}", partial)

        elif action in ("done", "finish", "stop"):
            # LLM returned done; optionally include result
            result = parsed.get("result", {})
            partial = {"order_id": order_id, "llm_history": llm_history, "trace": trace, "result": result}
            state_store.save_state(f"order:{order_id}", partial)
            return {"generated_order_id": order_id, "order_id": order_id, "trace": trace, "llm_history": llm_history, "result": result}

        elif action == "compensate":
            # LLM requested a compensation step (e.g., release stock). Interpret args to call relevant tool.
            comp_step = args.get("step")
            if comp_step == "release_inventory":
                # call inventory compensation endpoint if you have one (here we try update_order with cancelled)
                # placeholder: call update_order to 'cancel'
                t0 = time.time()
                tool_out = tool_update_order(order_id, "cancelled")
                t1 = time.time()
                trace.append({"step": "compensate_release", "out": tool_out, "latency": round(t1 - t0, 4)})
                llm_history.append({"role": "tool", "name": "compensate_release", "content": json.dumps(tool_out)})
                state_store.save_state(f"order:{order_id}", {"llm_history": llm_history, "trace": trace})
            else:
                # unknown compensate action; log and continue
                logging.warning("Unknown compensate action requested: %s", comp_step)
                llm_history.append({"role": "tool", "name": "compensate", "content": json.dumps({"error":"unknown_compensate"})})
                state_store.save_state(f"order:{order_id}", {"llm_history": llm_history, "trace": trace})

        else:
            # unknown action — stop to avoid infinite loop
            logging.warning("LLM suggested unknown action: %s -- stopping", action)
            llm_history.append({"role": "assistant", "content": json.dumps({"action":"done","result":{"reason":"unknown_action"}})})
            state_store.save_state(f"order:{order_id}", {"llm_history": llm_history, "trace": trace})
            return {"order_id": order_id, "trace": trace, "llm_history": llm_history, "result": {"reason": "unknown_action"}}

    # If we exit loop due to step limit, return partial
    logging.warning("Max steps reached for order_id=%s", order_id)
    state_store.save_state(f"order:{order_id}", {"llm_history": llm_history, "trace": trace})
    return {"generated_order_id": order_id, "order_id": order_id, "llm_history": llm_history, "trace": trace, "result": {"reason": "max_steps"}}

# ----------------------------
# LangGraph graph that uses orchestrator_node
# (We add a single 'orchestrator' node; the node itself runs the LLM/tool loop)
# ----------------------------
def build_agent_graph():
    if not langgraph_available:
        return None
    g = StateGraph()
    # Add a single python node that wraps orchestrator_node
    # In langgraph 1.0.2 you add a node via add_node(name, callable)
    g.add_node("orchestrator", orchestrator_node)
    g.add_edge(START, "orchestrator")
    compiled = g.compile()
    return compiled

_compiled_graph = build_agent_graph()
if _compiled_graph is None:
    # fallback "compiled" that simply wraps orchestrator_node invocation
    class _Fallback:
        def invoke(self, initial_state: Dict[str, Any], timeout: float = None):
            return orchestrator_node(initial_state)
    _compiled_graph = _Fallback()

# ----------------------------
# Public API
# ----------------------------
def invoke_agent(item: str, qty: int, order_id: str | None = None, timeout: float = None) -> Dict[str, Any]:
    oid = order_id or str(int(time.time() * 1000)) + "-" + str(os.getpid())
    initial_state = {"generated_order_id": oid, "order_id": oid, "item": item, "qty": qty}
    state_store.save_state(f"order:{oid}", initial_state)
    # call compiled graph
    try:
        out = _compiled_graph.invoke(initial_state, timeout=timeout) if timeout is not None else _compiled_graph.invoke(initial_state)
    except TypeError:
        out = _compiled_graph.invoke(initial_state)
    # compute total latency if not present
    if "total_latency" not in out:
        out["total_latency"] = round(sum(s.get("latency", 0) for s in out.get("trace", [])), 4) if out.get("trace") else 0.0
    # persist final state
    state_store.save_state(f"order:{oid}", out)
    return out

def load_state(order_id: str) -> Dict[str, Any]:
    return state_store.load_state(f"order:{order_id}")

# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    print("Demo invoking agent (LangGraph available=%s)" % langgraph_available)
    res = invoke_agent("laptop", 2)
    print(json.dumps(res, indent=2))
