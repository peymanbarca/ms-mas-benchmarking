## Experiment 1: 2 Agents
### Two LLM Agents with Atomic MongoDB Updates

This implementation uses the langgraph framework to define the two agents (Order and Inventory) and the langchain-openai library for LLM reasoning. The external database interaction is modeled using the asynchronous motor pattern to connect to MongoDB, which is crucial for non-blocking concurrent operations.