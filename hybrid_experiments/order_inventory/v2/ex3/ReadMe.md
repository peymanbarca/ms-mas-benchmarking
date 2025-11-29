# Experiment 3: two FastAPI microservices handling the order → inventory reservation cycle, each with its own MongoDB collection

Order Service: Receives order request, stores it in its DB, and calls Inventory Service.

Inventory Service: Checks stock and responds reserved / out_of_stock. Updates its own collection.

Each service is independent with its own DB collection.