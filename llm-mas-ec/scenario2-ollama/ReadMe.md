# Scenario 2:

##  asynchronous communication between agents, with RabbitMQ treat message send/receive as external events that trigger graph progression

    OrderAgent doesn’t directly call InventoryAgent in code, instead, it publishes a message (e.g., reserve_item_request) to RabbitMQ.
    
    Graph execution pauses (status=WAITING).

    InventoryAgent runs in its own process (or async worker), consumes that message, processes it, and then publishes back a reply (e.g., reserve_item_response).
    
    The LangGraph state machine is resumed when the response is received (like an event-driven transition).
    
    So LangGraph acts as the orchestrator, but RabbitMQ handles the async messaging layer.

## Only a fixed delay injected in response of reservation from inventory agent

## Run trial Experiments

### 1. Parallel Orders
    --> setenv OLLAMA_NUM_PARALLEL 10 to Ollama server be able to handle 10 requests concurrently
### 2. Sequential Orders
