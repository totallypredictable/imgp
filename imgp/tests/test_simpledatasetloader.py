from imgp.datasets.simpledatasetloader import SimpleDatasetLoader
from imgp.preprocessing.simpleprocessor import SimplePreprocessor
import numpy as np
import pytest


def test_load(img_paths):
    sp = SimplePreprocessor(32, 32)
    dl = SimpleDatasetLoader(preprocessors=[sp])
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
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert data.shape == (9, 32, 32, 3)
    assert labels.shape == (9,)
    assert len(labels) == len(expected)
    assert list(labels) == list(expected)


def test_load_without_preprocessors(img_paths):
    dl = SimpleDatasetLoader(preprocessors=None)
    with pytest.raises(Exception):
        dl.load(img_paths)


def test_load_verbose(img_paths):
    dl = SimpleDatasetLoader()
    with pytest.raises(Exception):
        dl.load(img_paths, 1)
