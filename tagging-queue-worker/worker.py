import os

import requests
import pika

PIKA_HOST = os.getenv("PIKA_HOST", "localhost")
PIKA_PORT = int(os.getenv("PIKA_PORT", "5672"))
PIKA_USER = os.getenv("PIKA_USER", "rmuser")
PIKA_PASS = os.getenv("PIKA_PASS", "rmpassword")

IMAGE_LINK_SERVICE_ENDPOINT = os.getenv("IMAGE_LINK_SERVICE_ENDPOINT", "")
POST_TAGS_ENDPOINT = os.getenv("POST_TAGS_ENDPOINT", "")
IMAGE_URL_PREDICT_ENDPOINT = os.getenv(
    "IMAGE_URL_PREDICT_ENDPOINT", "http://localhost:3000/download_and_predict"
)
IMAGE_URL_PREDICT_ENDPOINT_TIMEOUT = int(
    os.getenv("IMAGE_URL_PREDICT_ENDPOINT_TIMEOUT", 600)
)


def get_link(id: str, provide_sample: bool = True) -> str:
    if IMAGE_LINK_SERVICE_ENDPOINT:
        raise NotImplementedError()
    elif provide_sample:
        return "https://upload.wikimedia.org/wikipedia/en/thumb/0/05/Hello_kitty_character_portrait.png/220px-Hello_kitty_character_portrait.png"
    else:
        raise Exception("IMAGE_LINK_SERVICE_ENDPOINT is not specified.")


def post_tags(id: str, tags: dict[str], no_endpoint_ok: bool = True) -> str:
    if POST_TAGS_ENDPOINT:
        raise NotImplementedError()
    if not no_endpoint_ok:
        raise Exception("POST_TAGS_ENDPOINT is not specified.")


def get_tags_for_image_url(image_url: str) -> dict[str]:
    json_data = {"imgs_url": [image_url]}
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    request_url = IMAGE_URL_PREDICT_ENDPOINT
    r = requests.post(
        request_url,
        headers=headers,
        json=json_data,
        timeout=IMAGE_URL_PREDICT_ENDPOINT_TIMEOUT,
    )
    return r.json()[0]


def callback(ch, method, properties, body):
    try:
        message = body.decode()
        id = message
        print(f' [x] Received message: "{message}"', flush=True)
        link = get_link(str(message))
        print(f" [x] Received link: {link}", flush=True)
        tags = get_tags_for_image_url(link)
        print(f" [x] Got tags: {tags}", flush=True)
        post_tags(id, tags, no_endpoint_ok=True)
        print(f" [x] Posted tags: {id=}", flush=True)
        print(" [x] Done", flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except:
        print(" [x] Error!", flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    print(" [*] Connecting to RMQ...", flush=True)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=PIKA_HOST,
            port=PIKA_PORT,
            credentials=pika.PlainCredentials(username=PIKA_USER, password=PIKA_PASS),
            connection_attempts=2,
            retry_delay=5,
        )
    )
    print(" [*] Connected to RMQ!", flush=True)
    channel = connection.channel()

    channel.queue_declare(queue="task_queue", durable=True)
    print(" [*] Waiting for messages. To exit press CTRL+C", flush=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="task_queue", on_message_callback=callback)
    channel.start_consuming()
