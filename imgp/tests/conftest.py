import pytest
import os
import pathlib
import imgp

root = pathlib.Path(imgp.__file__).resolve().parent


@pytest.fixture()
def img_paths():
    return [
        os.path.join(root, "tests", "images", "cats", "cat-gfac16a4e0_640.jpg"),
        os.path.join(root, "tests", "images", "cats", "cat-ga7f8f27f4_640.jpg"),
        os.path.join(root, "tests", "images", "cats", "cat-g9f0ac3883_640.jpg"),
        os.path.join(root, "tests", "images", "frogs", "frog-g9c52de5c6_640.jpg"),
        os.path.join(root, "tests", "images", "frogs", "waters-g38a3c5715_640.jpg"),
        os.path.join(root, "tests", "images", "frogs", "frog-gdc7a17746_640.jpg"),
        os.path.join(root, "tests", "images", "giraffes", "giraffe-ge696dba15_640.jpg"),
        os.path.join(root, "tests", "images", "giraffes", "giraffe-g2122926f3_640.jpg"),
        os.path.join(root, "tests", "images", "giraffes", "giraffe-g9b3e5ee1f_640.jpg"),
    ]
