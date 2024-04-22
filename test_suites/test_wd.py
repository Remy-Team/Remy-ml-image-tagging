import os
import time
import csv

import numpy as np
from keras.preprocessing import image
from huggingface_hub import from_pretrained_keras


def load_keras_tagger_hf(tagger):
    model = from_pretrained_keras(tagger, compile=False)
    return model


def load_image(img_path, target_size):
    print(f"Loading {img_path}...")
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    return img


def indices_above_threshold(lst, threshold):
    return [index for index, element in enumerate(lst) if element > threshold]


def read_csv(file_path):
    # Create an empty dictionary to store column values
    data = {}
    # Open the CSV file
    with open(file_path, "r") as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Read the header row
        headers = next(csv_reader)

        # Initialize lists for each column
        for header in headers:
            data[header] = []

        # Read remaining rows
        for row in csv_reader:
            for i, value in enumerate(row):
                # Append value to corresponding column list
                data[headers[i]].append(value)
    return data


def load_proba_to_tag(csv_tags="tags.csv"):
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
    image_multiplier = 500  # Increase for benchmarking testing ~2im/s on well cooled cpu 8 core, recommend trying out onnx
    test_images_paths *= image_multiplier

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
    load_f = lambda x: load_image(x, input_shape)
    test_images = list(map(load_f, test_images))
    test_images = np.array(test_images)

    # Infer images
    predictions = model.predict(np.expand_dims(test_images[0], axis=0))
    start = time.time()
    predictions = model.predict(test_images, use_multiprocessing=True, workers=-1, batch_size=16)
    end = time.time()
    elapsed_time = end - start

    for img, predict in zip(test_images_paths, predictions):
        classes_nums = indices_above_threshold(predict, config["threshold"])
        classes = list(map(lambda x: proba_to_tag[x], classes_nums))

        print(f"IMAGE: {img}\nTAGS: {classes}\n")

    print(
        f"Predicted {len(test_images)} in {elapsed_time:.2f}s ({elapsed_time/len(test_images):.3f} s/im, {len(test_images)/elapsed_time:.3f} im/s)"
    )
