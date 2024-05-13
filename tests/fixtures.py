import os
import random

import pytest

from tests.settings import SAMPLE_IMAGES_DIR_PATH

def get_image_paths(images_dir = SAMPLE_IMAGES_DIR_PATH):
    files = os.listdir(images_dir)
    is_image = lambda filename: filename.lower().endswith(('.jpg', '.png', '.jpeg'))
    images = filter(is_image, files)
    images_paths = [os.path.join(images_dir, image) for image in images]
    assert len(images_paths) > 0, f"No images found in {images_dir=}"
    return images_paths

@pytest.fixture
def sample_images_paths():
    return get_image_paths()


@pytest.fixture
def sample_image_path(sample_images_paths):
    return random.choice(sample_images_paths)
