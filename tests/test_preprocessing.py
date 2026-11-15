from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import pytest
from preprocessing.images import decode_image, normalize, preprocess, PreprocessConfig


def test_resize_crop_normalization_and_no_mutation():
    image = np.full((200, 400, 3), 100, np.uint8)
    processed, blur = preprocess(image, PreprocessConfig(128, False, 1.5, (0, 0, 0.5, 1)))
    assert processed.shape == (128, 128, 3)
    assert processed[0, 0, 0] == 150 and image[0, 0, 0] == 100
    assert blur == 0
    assert normalize(processed).shape == (3, 128, 128)
    assert normalize(processed).dtype == np.float32
    assert normalize(processed).max() <= 1


def test_decode_rejects_invalid_and_applies_exif_orientation():
    with pytest.raises(ValueError):
        decode_image(b"not a photo")
    image = Image.new("RGB", (80, 40))
    exif = image.getexif()
    exif[274] = 6
    data = BytesIO()
    image.save(data, format="JPEG", exif=exif)
    assert decode_image(data.getvalue()).shape == (80, 40, 3)


def test_blur_and_invalid_crop():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (150, 150, 3), np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 4)
    assert preprocess(sharp, PreprocessConfig())[1] > preprocess(blurred, PreprocessConfig())[1]
    with pytest.raises(ValueError):
        preprocess(sharp, PreprocessConfig(crop=(0.5, 0, 0.1, 1)))
