import os
import csv

import numpy as np
import keras
import typing as t
from keras.preprocessing import image
import bentoml
from PIL.Image import Image as PILImage


def read_csv(file_path):
    data = {}
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for header in headers:
            data[header] = []
        for row in csv_reader:
            for i, value in enumerate(row):
                data[headers[i]].append(value)
    return data


def load_proba_to_tag(csv_tags="tags.csv"):
    read_columns = read_csv(csv_tags)
    return {
        k: (v[0], v[1])
        for k, v in enumerate(zip(read_columns["name"], read_columns["category"]))
    }


def indices_above_threshold(lst, threshold):
    return [index for index, element in enumerate(lst) if element > threshold]


def probs_to_tags_rating(probs, threshold: float, proba_to_tag: dict) -> list:
    """Based on predicted probas, return the verdict, whether
    image is nsfw or not"""
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


# Preprocessing
def preprocess_img(img, img_dim):
    img = img.resize((img_dim, img_dim))
    arr = np.array(img, dtype=np.float64)
    arr = arr[..., :3]
    # arr /= 255
    return arr


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
    model_ref = bentoml.keras.get(f"{MODEL_TAG}:latest")

    def __init__(self) -> None:
        self.proba_to_tag = load_proba_to_tag(TAGS_FILE)
        self.model: keras.Model = self.model_ref.load_model()
        self.threshold = THRESHOLD
        self.img_dim = IMG_DIM
        print(f"Service initialized successfully")

    @bentoml.api(batchable=True, max_batch_size=BATCH_SIZE, max_latency_ms=BATCH_TIMEOUT, batch_dim=0)
    def predict(self, imgs: t.List[PILImage]) -> t.List[dict]:
        load_f = lambda x: preprocess_img(x, IMG_DIM)
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