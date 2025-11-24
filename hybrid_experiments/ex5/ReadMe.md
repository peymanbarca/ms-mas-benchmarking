## Experiment 5: Two Agents and Two Microserivces

This experiment, Experiment 5: Two Agents vs. Two Microservices, models the most complex distributed environment: two specialized LLM agents coordinating a transactional workflow, with each agent responsible for interacting with a dedicated, deterministic microservice (Order Service and Inventory Service).

The primary focus is on managing Protocol Consistency and ensuring the Temporal Persistence (commitment) of the combined plan across four distributed components, often using the SAGA pattern for compensation logic.

### Implementation: Distributed System Architecture
The system consists of three required files that must run concurrently:

    inventory_service.py (Resource Manager): Performs atomic stock updates on MongoDB.
    
    order_service.py (Local Transaction Manager): Manages the order's status life cycle on MongoDB.
    
    fulfillment_system.py (Multi-Agent Coordinator): The LangGraph system coordinating the two LLM agents and managing the communication flow between the two microservices.