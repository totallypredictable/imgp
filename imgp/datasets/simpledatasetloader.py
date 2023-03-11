from imgp.preprocessing.simpleprocessor import SimplePreprocessor
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    """
    The SimpleDatasetLoader() loads in images from a list of given paths,
    applying any preprocessors provided. All of the images will be returned
    at once, so make sure the images can actually fit into memory.

    Parameters
    ----------
    preprocessors: list, default=None
        The list of preprocessors to be applied to the images.

    Methods
    -------
    load:
        Load the images from the given paths and preprocess them given the
        preprocessors.
    """

    def __init__(self, preprocessors: list[SimplePreprocessor] | None = None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are NOne, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(
        self, imagePaths: list[str], verbose: int = -1
    ) -> tuple[np.ndarray, np.ndarray]:
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming that
            # our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if len(self.preprocessors) > 0:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector" by updating
            # the data lsit followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every "verbose" images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        return (np.array(data), np.array(labels))
