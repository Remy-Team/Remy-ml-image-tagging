"""Raw benchmarking of the model"""

import os
import time
import csv
from functools import partial

import numpy as np
from keras.preprocessing import image
from huggingface_hub import from_pretrained_keras

def load_keras_tagger_hf(tagger):
    """Load tagger from huggingface"""
    return from_pretrained_keras(tagger, compile=False)


def load_image(img_path, target_size):
    """Loads and preprocesses image"""
    print(f"Loading {img_path}...")
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    return img


def indices_above_threshold(lst, threshold):
    """Converts probas to tags indexes"""
    return [index for index, element in enumerate(lst) if element > threshold]


def read_csv(file_path):
    """Reads a csv file columnwise"""
    data = {}
    with open(file_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        for header in headers:
            data[header] = []
        for row in csv_reader:
            for i, value in enumerate(row):
                data[headers[i]].append(value)
    return data


def load_proba_to_tag(csv_tags="tags.csv"):
    """Gets a dictionary of index to category"""
    read_columns = read_csv(csv_tags)
    return {
        k: (v[0], v[1])
        for k, v in enumerate(zip(read_columns["name"], read_columns["category"]))
    }


if __name__ == "__main__":
    config = {
        "tagger": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "threshold": 0.5,
        "test_images_dir": "data/sample_images",
        "proba_to_tag_csv": "data/tags.csv",
    }

    test_images_paths = [
        os.path.join(config["test_images_dir"], img_name)
        for img_name in os.listdir(config["test_images_dir"])
        if img_name.endswith(".jpg")
    ]
    # Increase for benchmarking testing ~2im/s on well
    # cooled cpu 8 core, recommend trying out onnx
    IMAGE_MULTIPLIER = 500
    test_images_paths *= IMAGE_MULTIPLIER

    test_images = test_images_paths.copy()
    print(f"Test images count {len(test_images)}")
    assert len(test_images) != 0

    print("Loading model...")

    model = load_keras_tagger_hf(config["tagger"])
    proba_to_tag = load_proba_to_tag(config["proba_to_tag_csv"])
    input_shape = model.input_shape[1:3]
    output_shape = model.output_shape

    print(f"Model input shape: {input_shape}")
    print(f"Model output shape: {output_shape}")

    # Prepare images
    load_f = partial(load_image, target_size=input_shape)
    test_images = list(map(load_f, test_images))
    test_images = np.array(test_images)

    # Infer images
    predictions = model.predict(np.expand_dims(test_images[0], axis=0))
    start = time.time()
    predictions = model.predict(
        test_images, use_multiprocessing=True, workers=-1, batch_size=16
    )
    end = time.time()
    elapsed_time = end - start

    for img, predict in zip(test_images_paths, predictions):
        classes_nums = indices_above_threshold(predict, config["threshold"])
        classes = list(map(lambda x: proba_to_tag[x], classes_nums))
        print(f"IMAGE: {img}\nTAGS: {classes}\n")
    print(
        f"Predicted {len(test_images)} in {elapsed_time:.2f}s "
        f"({elapsed_time/len(test_images):.3f} s/im or "
        f"{len(test_images)/elapsed_time:.3f} im/s)"
    )
