from tensorflow.keras.utils import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apple the keras utility function taht correctly rearranges the
        # dimensions of the imeage
        return img_to_array(image, data_format=self.dataFormat)
