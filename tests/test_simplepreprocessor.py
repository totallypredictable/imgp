import numpy as np
from imgp.preprocessing import SimplePreprocessor
from PIL import Image
import cv2
import numpy.testing as npt


def create_in_memory_image():
    img = Image.new("RGB", (800, 1280), (255, 255, 255))
    return np.array(img)


def test_preprocess():
    sp = SimplePreprocessor(32, 32)
    image = create_in_memory_image()
    expected_img = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    returned_img = sp.preprocess(image)
    npt.assert_array_equal(expected_img, returned_img)
