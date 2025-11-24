## Experiment 4: Two Agents and One Microserivce

This experiment, Two Agents vs. One Microservice, is designed to test how non-deterministic coordination between two specialized LLM agents (Order Agent and Inventory Agent) impacts the reliable execution of a transaction against a single, deterministic source of truth (the FastAPI Microservice).
The primary challenge is managing the concurrent access and the handoff protocol between agents to avoid issues like Phantom Consistency and Protocol Failure.

Implementation: Microservice and Agent System
The system is composed of two required files that must run concurrently:

    1. inventory_microservice.py (Deterministic Tool): The authoritative transactional system (reused from Experiment 2, providing atomic MongoDB operations).
    2. fulfillment_system.py (Coordinating Agents): The LangGraph system coordinating the two LLM agents and the shared microservice tool.