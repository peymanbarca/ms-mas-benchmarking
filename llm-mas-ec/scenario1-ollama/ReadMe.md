# Scenario 1:

##  direct function-call chaining (sync communication with internal state passing of a mutable state object) using LangGraph synchronous state machine
    # OrderAgent (INIT order)
    #    → InventoryAgent (decide reservation)
    #    → OrderAgent (finalize order status)
    #    → END

## Only a fixed delay injected in response of reservation from inventory agent

## Run trial Experiments

### 1. Parallel Orders
    --> setenv OLLAMA_NUM_PARALLEL 10 to Ollama server be able to handle 10 requests concurrently
### 2. Sequential Orders
