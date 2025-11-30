# Experiment 2: One LLM Agent vs One FastAPI Microservice implementation (1 LLM Call per trial)

The agent is LLM-based (same partial-state), uses a small LangGraph-like wrapper and calls the microservice over HTTP to reserve inventory. It persists its lightweight state in Redis and the order record in MongoDB (REAL or MOCK modes preserved).

The microservice is a FastAPI app that implements an atomic POST /reserve endpoint (using MongoDB find_one_and_update with stock >= qty) and POST /health and POST /reset for test setup. It supports artificial delay and drop-rate injection via environment variables.

A runner harness is provided that runs sequential/parallel trials, collects per-trial latency and final-state metrics, and saves results for plotting.