# Experiment 4: LLM Orchestrator Agent + Microservices as Tools


Evaluate how an LLM-based reasoning agent can orchestrate a workflow across existing microservices (Order, Inventory) and measure:

Latency added by reasoning + tool invocation

Consistency and correctness of multi-step orchestration

Robustness under injected failures

Comparison vs pure MSA orchestration (no LLM)

1. Architecture
┌─────────────────────────────────────────────┐
│               LLM Orchestrator              │
│  (reasoning, planning, calling tools)       │
└───────────────┬─────────────────────────────┘
                │ Tools API
 ┌──────────────┴──────────────┐
 │                             │
▼                              ▼
Inventory Microservice       Order Microservice
(check stock)                 (create order)


The LLM agent receives a high-level request:

“Place an order for 2 laptops”


It decides:

Call check_inventory(item="laptop", qty=2)

If success → call create_order(...)

Return final reasoning trace + outcome

--------------------
## Includes: 

an LLM-based orchestrator agent implemented with LangGraph (if available) and with a small fallback so the code runs even without LangGraph / an LLM; the agent reasons (or simulates reasoning) then calls microservices as tools;

two FastAPI microservices: Inventory and Order, each with its own MongoDB collection;

a runner that triggers parallel requests to the agent and records per-request latency and final outcome;

## Mechanism of agent

Iterative decision loop — LLM receives a prompt that includes all prior messages and tool outputs (llm_history). The LLM returns a structured JSON {"action":..., "args":...} which the agent executes. The tool output is appended to llm_history and fed back to the LLM, enabling the LLM to choose the next action based on live tool results.

State tracking — Redis is used to persist state after each step. Partial dicts are saved so you can resume or inspect progress.

Safety & fallback — if no LLM is configured, call_llm_text returns deterministic actions so you can test the flow.