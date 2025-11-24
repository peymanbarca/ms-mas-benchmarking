## Experiment 3: One Microservice & another Microservice

This experiment, Experiment 3: Two Microservices Interaction, acts as a crucial control group for agent-based system reliability. Here, two deterministic FastAPI microservices—the Order Service and the Inventory Service—interact directly using the SAGA pattern to maintain transactional consistency.

Since no Large Language Model (LLM) is involved, this experiment isolates extrinsic, systemic non-determinism (network latency, hardware variability, distributed state synchronization) from intrinsic LLM non-determinism (reasoning, tool argument generation). If inconsistency occurs, the root cause is a traditional software failure (race condition, network timeout, forgotten rollback) rather than stochastic AI behavior.
Implementation: Order and Inventory Microservices

These two services must be run concurrently for the experiment to function.