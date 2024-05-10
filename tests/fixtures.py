import os
import random

import pytest

from settings import SAMPLE_IMAGES_DIR_PATH

@pytest.fixture
def sample_images_paths():
    files = os.listdir(SAMPLE_IMAGES_DIR_PATH)
    is_image = lambda filename: filename.lower().endswith(('.jpg', '.png', '.jpeg'))
    images = filter(is_image, files)
    images_paths = [os.path.join(SAMPLE_IMAGES_DIR_PATH, image) for image in images]
    assert len(images_paths) > 0, f"No images found in {SAMPLE_IMAGES_DIR_PATH=}"
    return images_paths

@pytest.fixture
def sample_image_path(sample_images_paths):
    return random.choice(sample_images_paths)
