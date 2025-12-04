## Experiment 1: Two Agents (3 LLM Calls + 3 DB Write per trial)

### Two LLM Agents with Atomic MongoDB Updates

This implementation uses the langgraph framework to define the two agents (Order and Inventory),  and MongoDB database and redis for state persistence


    order_agent(state, db_ag) and inventory_agent(state, db_ag) call the LLM and return partial dicts like {"status":"INIT", "forward": True} or {"status":"reserved"}.
    
    build_graph(db_ag) creates a StateGraph that routes based on state["status"].
    
    run_trial() runs a single trial and returns metrics (latency, CPU, memory, final state).
    
    sequential_trials() and parallel_trials() run many trials and log results.
    
    Inject delay (seconds) and drop_rate (%) via global variables or parameters.