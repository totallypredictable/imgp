from imgp.datasets.simpledatasetloader import SimpleDatasetLoader
from imgp.preprocessing.simpleprocessor import SimplePreprocessor
import numpy.testing as npt
import numpy as np


def test_load(img_paths):
    sp = SimplePreprocessor(32, 32)
    dl = SimpleDatasetLoader([sp])
    data, labels = dl.load(img_paths)
    expected = np.array(
        [
            "cats",
            "cats",
            "cats",
            "frogs",
            "frogs",
            "frogs",
            "giraffes",
            "giraffes",
            "giraffes",
        ]
    )
    assert data.shape == (9, 32, 32, 3)
    assert labels.shape == (9,)
    assert len(labels) == len(expected)
    assert list(labels) == list(expected)
