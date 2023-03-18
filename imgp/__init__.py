import pathlib

import imgp

PACKAGE_ROOT = pathlib.Path(imgp.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

name = "imgp"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()

__all__ = [
    "callbacks",
    "datasets",
    "nn",
    "preprocessing",
]
