"""Bentoml service for tagging code"""

import os
import csv
from functools import partial
import typing as tp
import io
import asyncio
import tempfile


import requests
import numpy as np
import keras
import bentoml
from PIL import Image


def read_csv(file_path):
    """Reads csv columnwise"""
    data = {}
    with open(file_path, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for header in headers:
            data[header] = []
        for row in csv_reader:
            for i, value in enumerate(row):
                data[headers[i]].append(value)
    return data


def load_proba_to_tag(csv_tags="tags.csv"):
    """Load dictionary to convert indexes to tags"""
    read_columns = read_csv(csv_tags)
    return {
        k: (v[0], v[1])
        for k, v in enumerate(zip(read_columns["name"], read_columns["category"]))
    }


def indices_above_threshold(lst, threshold):
    """Returns a list of indexes based on probas"""
    return [index for index, element in enumerate(lst) if element > threshold]


def probs_to_tags_rating(probs, threshold: float, proba_to_tag: dict) -> list:
    """Parses probas and returns tags and Image.Image rating"""
    classes_nums = indices_above_threshold(probs, threshold)
    classes = list(map(lambda x: proba_to_tag[x], classes_nums))
    tags = []
    rating = ""
    for c in classes:
        name, tag_type = c
        if tag_type == "9":
            rating = name
        else:
            tags.append(name)
    return tags, rating


def preprocess_img(img, img_dim):
    """Preprocesses an Image.Image before inference"""
    img = img.resize((img_dim, img_dim))
    arr = np.array(img, dtype=np.float64)
    arr = arr[..., :3]
    return arr


async def download_image(url: str) -> Image.Image | str:
    with tempfile.SpooledTemporaryFile(max_size=1e9) as buffer:
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024):
                    downloaded += len(chunk)
                    buffer.write(chunk)
                buffer.seek(0)
                return Image.open(io.BytesIO(buffer.read()))
            else:
                raise Exception(f"Non-200 status code: {r.status_code}")
        except Exception as e:
            text = f"Failed to download the Image.Image. Reason: {e=}"
            print(text)
            return text


async def urls_to_imgs(urls: tp.List[str]) -> tp.List[Image.Image | str]:
    futures = list(map(download_image, urls))
    return await asyncio.gather(*futures)


THRESHOLD = 0.5
MODEL_TAG = "wd14-remy"
TAGS_FILE = "tags.csv"
IMG_DIM = 448
WORKERS = int(os.getenv("NUM_WORKERS", "1"))
CPUS_PER_WORKER = os.getenv("CPUS_PER_WORKER", "1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "20000"))


@bentoml.service(
    workers="cpu_count" if WORKERS == -1 else WORKERS,
    resources={"cpu": CPUS_PER_WORKER, "memory": "2Gi"},
)
class ImageTagging:
    """BentoML Service class for tagging Image.Images"""

    model_ref = bentoml.keras.get(f"{MODEL_TAG}:latest")

    def __init__(self) -> None:
        self.proba_to_tag = load_proba_to_tag(TAGS_FILE)
        self.model: keras.Model = self.model_ref.load_model()
        self.threshold = THRESHOLD
        self.img_dim = IMG_DIM
        print("Service initialized successfully")

    def _infer_imgs_list(self, imgs: tp.List[Image.Image]) -> tp.List[dict]:
        """Predicts tags for an Image.Image"""
        load_f = partial(preprocess_img, img_dim=IMG_DIM)
        imgs = list(map(load_f, imgs))
        imgs = np.array(imgs)

        # Inference
        preds = self.model.predict(imgs, verbose=False, batch_size=BATCH_SIZE)
        results = []
        for probs in preds:
            # Tagger specific
            tags, rating = probs_to_tags_rating(
                probs, threshold=self.threshold, proba_to_tag=self.proba_to_tag
            )
            result = {
                "message": "Success!",
                "rating": rating,
                "tags": tags,
            }
            results.append(result)
        return results

    @bentoml.api(
        batchable=True,
        max_batch_size=BATCH_SIZE,
        max_latency_ms=BATCH_TIMEOUT,
        batch_dim=0,
    )
    def predict(self, imgs: tp.List[Image.Image]) -> tp.List[dict]:
        """For testing purposes"""
        return self._infer_imgs_list(imgs)

    def process_images_and_errors(self, imgs_and_errors: tp.List[Image.Image | str]):
        images = [im for im in imgs_and_errors if isinstance(im, Image.Image)]
        results_images = self._infer_imgs_list(images) if images else []
        results_images_i = 0
        results = []
        for el in imgs_and_errors:
            if isinstance(el, str):
                result = {
                    "message": el,
                    "rating": None,
                    "tags": None,
                }
            elif isinstance(el, Image.Image):
                result = results_images[results_images_i]
                results_images_i += 1
            else:
                result = {}
            results.append(result)
        return results

    @bentoml.api(
        batchable=True,
        max_batch_size=BATCH_SIZE,
        max_latency_ms=BATCH_TIMEOUT,
        batch_dim=0,
    )
    async def download_and_predict(self, imgs_url: tp.List[str]) -> tp.List[dict]:
        imgs_and_errors = await urls_to_imgs(imgs_url)
        return self.process_images_and_errors(imgs_and_errors)
