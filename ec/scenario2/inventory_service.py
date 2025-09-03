import time, json, random, pika, pymongo, uuid, threading, queue
from fastapi import FastAPI

app = FastAPI(title="Inventory Service")

mongo = pymongo.MongoClient("mongodb://user:pass1@localhost:27017/")
db = mongo["retail"]
inventory = db["inventory"]

consumer_connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=pika.PlainCredentials("guest", "guest")
    )
)
consumer_channel = consumer_connection.channel()
consumer_channel.queue_declare(queue="inventory_queue")


# Queue for passing delivery_tags back to the consumer thread for ack
ack_queue = queue.Queue()

DROP_RATE = 0.2  # simulate reply message drop


def process_request(data, props, delivery_tag):
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

    # todo: Decide whether to drop the reply
    if random.random() < DROP_RATE:
        print(f"[DROP] Not replying to order {data['order_id']}")
    else:
        reply_conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host="localhost",
                port=5672,
                credentials=pika.PlainCredentials("guest", "guest")
            )
        )
        reply_channel = reply_conn.channel()
        reply_channel.basic_publish(
            exchange="",
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=json.dumps(response),
        )
        reply_conn.close()
        print(f"Responded to order {data['order_id']} with {status}")

    ack_queue.put(delivery_tag)


def on_request(ch, method, props, body):
    data = json.loads(body)
    # Spawn worker thread for processing
    threading.Thread(
        target=process_request,
        args=(data, props, method.delivery_tag),
        daemon=True
    ).start()


def ack_pending():
    """Called periodically in main thread to ack completed deliveries"""
    while not ack_queue.empty():
        delivery_tag = ack_queue.get_nowait()
        consumer_channel.basic_ack(delivery_tag=delivery_tag)


consumer_channel.basic_qos(prefetch_count=10)
consumer_channel.basic_consume(queue="inventory_queue", on_message_callback=on_request)

print("InventoryService is up and waiting for requests...")
while True:
    consumer_connection.process_data_events(time_limit=1)
    ack_pending()

