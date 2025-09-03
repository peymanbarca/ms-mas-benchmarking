import time
import uuid, json, pika, pymongo, threading
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Order Service")

# Mongo setup
mongo = pymongo.MongoClient("mongodb://user:pass1@localhost:27017/")
db = mongo["retail"]
orders = db["orders"]

# RabbitMQ setup (persistent connection for consumer)
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
lock = threading.Lock()

def on_response(ch, method, props, body):
    correlation_id = props.correlation_id
    with lock:
        if correlation_id in responses:
            responses[correlation_id] = json.loads(body)

channel.basic_consume(
    queue=callback_queue,
    on_message_callback=on_response,
    auto_ack=True,
)

# Run consumer in background thread
def consume_loop():
    channel.start_consuming()

threading.Thread(target=consume_loop, daemon=True).start()

@app.post("/order")
def create_order(request: dict):
    item: str = request["item"]
    qty: int = request["qty"]
    order_id = str(uuid.uuid4())
    orders.insert_one({"_id": order_id, "item": item, "qty": qty, "status": "INIT"})

    correlation_id = str(uuid.uuid4())
    with lock:
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
        body=json.dumps({"order_id": order_id, "item": item, "qty": qty}),
    )
    connection_publish.close()

    # Wait for response (with timeout)
    timeout = 60
    start = time.time()
    while True:
        with lock:
            res = responses.get(correlation_id)
        if res is not None:
            with lock:
                responses.pop(correlation_id, None)
            orders.update_one({"_id": order_id}, {"$set": {"status": res["status"]}})
            return {"order_id": order_id, "reservation_status": res["status"]}
        if time.time() - start > timeout:
            return {"order_id": order_id, "reservation_status": "timeout"}
        time.sleep(0.1)
