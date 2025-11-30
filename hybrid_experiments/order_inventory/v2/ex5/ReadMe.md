# Experiment 5: Two Agents + Two Microservices (2 LLM Call per trial)

Two FastAPI microservices using MongoDB:

    inventory_service.py — /check, /reserve, /reset
    
    order_service.py — /create, /update, /reset

Two LangGraph 1.0.2–style agents in a single module agents_exp5.py:
    
    Graph orchestration of two agents using LangGraph (StateGraph).

        Inventory Agent (calls Inventory microservice)
        
        Order Agent (orchestrator: asks LLM for next action, invokes Inventory Agent, then calls Order microservice)

    LLM reasoning per node using Ollama Qwen2.

    Tool calls:

        Order Service: create order, update order status

        Inventory Service: check & reserve stock


    Both agents use Redis for state tracking (Option B partial-state returns).

A parallel runner that:

    Runs N_TRIALS parallel calls to the Order Agent (the entry point)
    
    Saves per-trial results and computes batch-level metrics (avg latency, consistency error)

    JSON output includes full orchestration trace for plotting latency vs consistency.