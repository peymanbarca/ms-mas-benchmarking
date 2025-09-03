import time, json, random, pika, pymongo, uuid
from fastapi import FastAPI

app = FastAPI(title="Inventory Service")

# Mongo setup
mongo = pymongo.MongoClient("mongodb://user:pass1@localhost:27017/")
db = mongo["retail"]
inventory = db["inventory"]


# RabbitMQ setup
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=pika.PlainCredentials("guest", "guest")
    )
)
channel = connection.channel()
channel.queue_declare(queue="inventory_queue")


def on_request(ch, method, props, body):
    data = json.loads(body)
    item, qty = data["item"], data["qty"]

    stock = inventory.find_one({"item": item}) or {"item": item, "stock": 10}
    if stock["stock"] >= qty:
        reservation_id = str(uuid.uuid4())
        inventory.update_one({"item": item}, {"$inc": {"stock": -qty}}, upsert=True)
        status = "reserved"

        # Simulate delay
        time.sleep(3)

    else:
        status = "out_of_stock"
        reservation_id = None

    response = {"order_id": data["order_id"], "reservation_id": reservation_id, "status": status}

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps(response),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="inventory_queue", on_message_callback=on_request)

print("InventoryService waiting for requests...")
channel.start_consuming()
