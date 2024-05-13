from functools import partial
import random

from locust import HttpUser, task, between

from tests.fixtures import get_image_paths

class ImageTaggingUser(HttpUser):
    wait_time = between(1, 3)

    def _send_infer_request(self, image_path):
        """Sends a single image for inference to ML service"""
        headers = {'accept': 'application/json'}
        files = {'imgs': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')}
        self.client.post('/predict', headers=headers, files=files, timeout=300)

    @task
    def send_single_image_for_inference(self):
        """Sends a single image for inference to ML service"""
        image_path = random.choice(get_image_paths())
        self._send_infer_request(image_path)

    @task
    def send_several_images_for_inference_sequentially(self):
        """Sends several images in a row for inference to ML service"""
        get_image_path = partial(random.choice, seq=get_image_paths())
        for image_path in [get_image_path() for _ in range(10)]:
            self._send_infer_request(image_path)



