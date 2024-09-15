import os

import numpy as np
from setuptools import Extension, setup

include_dirs = [
    np.get_include(),
    "\\".join((os.path.dirname(os.path.realpath(__file__)), "include")),
]

setup(
    ext_modules=[
        Extension(
            "transforms84.transforms",
            sources=["include/transforms.c"],
            include_dirs=include_dirs,
        ),
        Extension(
            "transforms84.distances",
            sources=["include/distances.c"],
            include_dirs=include_dirs,
        ),
        Extension(
            "transforms84.helpers",
            sources=["include/helpers.c"],
            include_dirs=include_dirs,
        ),
    ],
)
