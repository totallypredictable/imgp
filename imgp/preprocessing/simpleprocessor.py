import cv2
import numpy.typing as npt
import numpy as np


class SimplePreprocessor:
    def __init__(self, width: int, height: int, inter: int = cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width: int = width
        self.height: int = height
        self.inter = inter

    def preprocess(self, image: npt.NDArray[np.uint8]):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
