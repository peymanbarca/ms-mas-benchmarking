## Experiment 2: one Agent & one Mircoservice


This experiment, One Agent vs. One Microservice, focuses on the fundamental challenge of ensuring Protocol Consistency and preventing Hallucinated Side-Effects when a non-deterministic reasoning engine (the LLM Agent) interfaces with a deterministic transactional system (the FastAPI Microservice/MongoDB).
The agent is responsible for the overall plan (Reserve $\rightarrow$ Confirm $\rightarrow$ End), but the microservice is the sole source of truth and execution. Failures now originate from the LLM's inability to correctly interact with the deterministic tool.
I. Implementation: Microservice and Agent Code
The system is split into two components that must run concurrently:
    1. inventory_microservice.py (The Deterministic Tool): A FastAPI service using motor to perform atomic, transactional updates on MongoDB.
    2. fulfillment_agent.py (The Planning Engine): A single LangGraph agent that uses LLM reasoning to decide which API endpoint to call next. It uses an asynchronous HTTP client (httpx) to call the microservice.Order and Inventory) and the langchain-openai library for LLM reasoning. The external database interaction is modeled using the asynchronous motor pattern to connect to MongoDB, which is crucial for non-blocking concurrent operations.