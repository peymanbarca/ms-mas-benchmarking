# Scenario 1:

##  direct function-call chaining (sync communication)
    # OrderAgent (INIT order)
    #    → InventoryAgent (decide reservation)
    #    → OrderAgent (finalize order status)
    #    → END

## Only a fixed delay injected in response of reservation from inventory agent

## Run trial Experiments

### 1. Parallel Orders
### 2. Sequential Orders
