import os
import typing as tp

from fastapi import FastAPI
import pika

PIKA_HOST = os.getenv("PIKA_HOST", "localhost")
PIKA_PORT = os.getenv("PIKA_PORT", "5672")
PIKA_USER = os.getenv("PIKA_USER", "rmuser")
PIKA_PASS = os.getenv("PIKA_PASS", "rmpassword")

app = FastAPI(title="tagging-queue")

print(" [*] Connecting to RMQ...", flush=True)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host=PIKA_HOST,
        port=PIKA_PORT,
        credentials=pika.PlainCredentials(
            username=PIKA_USER, 
            password=PIKA_PASS
        ),
        connection_attempts=2,
        retry_delay=5,
    )
)
channel = connection.channel()
channel.queue_declare(queue="task_queue", durable=True)
print(" [*] Connected to RMQ!", flush=True)

def _send_id_to_queue(id: str):
    message = id
    channel.basic_publish(
        exchange="",
        routing_key="task_queue",
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        ),
    )
    print(f" [x] Sent {message=} to queue")

def _queue_id(id) -> dict:
    try:
        _send_id_to_queue(id)
        return {"message": "Success!", "id": id}
    except Exception as e:
        return {"message": f"Error: {e}", "id": None}

@app.post("/queue/image_by_id")
def _queue_image_by_id(id: str):
    return _queue_id(id)

@app.post("/queue/images_by_id_list")
def _queue_images_by_id_list(ids: tp.List[str]):
    return list(map(_queue_id, ids))
    