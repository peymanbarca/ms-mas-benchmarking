import time
import uuid, json, pika, pymongo, threading
from fastapi import FastAPI

app = FastAPI(title="Order Service")

mongo = pymongo.MongoClient("mongodb://user:pass1@localhost:27017/")
db = mongo["retail"]
orders = db["orders"]


connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=pika.PlainCredentials("guest", "guest")
    )
)
channel = connection.channel()

# Create exclusive reply queue for this service
result = channel.queue_declare(queue="", exclusive=True)
callback_queue = result.method.queue

responses = {}
# lock = threading.Lock()


def on_response(ch, method, props, body):
    """Called whenever inventory sends a response"""
    res = json.loads(body)
    order_id = res["order_id"]
    status = res["status"]
    orders.update_one({"_id": order_id}, {"$set": {"status": status}})
    print(f"[OrderService] Updated order {order_id} -> {status}")


channel.basic_consume(
    queue=callback_queue,
    on_message_callback=on_response,
    auto_ack=True,
)


# Run consumer in background thread
def consume_loop():
    channel.start_consuming()


def consume_forever():
    """Run consumer in background thread"""
    print("[OrderService] Waiting for inventory responses...")
    while True:
        connection.process_data_events(time_limit=1)


threading.Thread(target=consume_forever, daemon=True).start()


@app.post("/order")
def create_order(request: dict):
    item: str = request["item"]
    qty: int = request["qty"]
    delay: int = request["delay"]
    order_id = str(uuid.uuid4())
    orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT"})

    correlation_id = str(uuid.uuid4())
    #with lock:
    responses[correlation_id] = None

    # Open a fresh connection for publishing
    connection_publish = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="localhost",
            port=5672,
            credentials=pika.PlainCredentials("guest", "guest")
        )
    )
    channel_publish = connection_publish.channel()

    channel_publish.basic_publish(
        exchange="",
        routing_key="inventory_queue",
        properties=pika.BasicProperties(
            reply_to=callback_queue,
            correlation_id=correlation_id,
        ),
        body=json.dumps({"order_id": order_id, "item": item, "qty": qty, "delay": delay}),
    )
    connection_publish.close()

    # Return immediate response, status will be updated later in background
    return {"order_id": order_id, "msg": "Reservation Requested"}
