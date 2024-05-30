"""Tests for ML tagging service deployed via BentoML"""
import asyncio

import aiohttp
import requests

from tests.settings import SERIVCE_ENDPOINT_URL

def requests_infer_tags_from_image(image_path) -> requests.Response:
    """Sends a single image for inference to ML service"""
    headers = {"accept": "application/json"}
    with open(image_path, "rb") as f:
        files = {
            "imgs": ("image.jpg", f, "image/jpeg"),
        }
        request_url = f"{SERIVCE_ENDPOINT_URL}/predict"
        response = requests.post(request_url, headers=headers, files=files)
    return response


async def aiohttp_infer_tags_from_image(image_path):
    """Sends a single image for inference to ML service asynchronously"""
    headers = {"accept": "application/json"}
    data = aiohttp.FormData()
    data.add_field(
        "imgs", open(image_path, "rb"), filename="image.jpg", content_type="image/jpeg"
    )
    async with aiohttp.ClientSession(base_url=SERIVCE_ENDPOINT_URL) as session:
        async with session.post(
            "/predict", headers=headers, data=data, timeout=180
        ) as response:
            return await response.json()


def requests_get_livez():
    headers = {"accept": "*/*"}
    request_url = f"{SERIVCE_ENDPOINT_URL}/livez"
    response = requests.get(request_url, headers=headers)
    return response


class TestTaggingService:
    def test_service_is_up(self):
        """
        Checks if service is ready by getting /livez
        Expectation: 200 means service is ready
        """
        response = requests_get_livez()
        assert response.status_code == 200, response.content

    @staticmethod
    def _assert_correct_prediction_response_json(data: dict):
        assert isinstance(data, list), data
        assert len(data) == 1
        prediction = data[0]
        required_keys = ["message", "rating", "tags"]
        assert set(prediction.keys()) == set(required_keys)
        assert prediction["message"] == "Success!"
        assert isinstance(prediction["rating"], str)
        assert isinstance(prediction["tags"], list)

    @staticmethod
    def _assert_correct_single_prediction_response_requests(response):
        status_code = response.status_code
        assert status_code == 200, response.content
        data = response.json()
        TestTaggingService._assert_correct_prediction_response_json(data)

    def test_single_image_inference(self, sample_image_path):
        """
        Checks if service is working
        when asking to inference a single image
        Expectation: 200 status code and correct response format

        Example response:
        [
            {
                "message": "Success!",
                "rating": "general",
                "tags": [
                "comic",
                "english_text",
                "no_humans",
                "text_focus"
                ]
            }
        ]
        """
        response = requests_infer_tags_from_image(sample_image_path)
        TestTaggingService._assert_correct_single_prediction_response_requests(response)

    async def test_parallel_image_inference(self, sample_images_paths):
        """
        Checks if service is working
        when asking to inference multiple images using
        diffrent requests
        Expectation: 200 status code and correct response format
        """
        assert len(sample_images_paths) >= 5, "Put more images in test folder (min 5)"
        image_paths = sample_images_paths[:5] * 1
        futures = []
        for image_path in image_paths:
            future = aiohttp_infer_tags_from_image(image_path)
            futures.append(future)

        response_jsons = await asyncio.gather(*futures)
        for response_json in response_jsons:
            TestTaggingService._assert_correct_prediction_response_json(response_json)
