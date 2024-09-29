import os
import sys
from setuptools import Extension, setup

include_dirs = [
    np.get_include(),
    "\\".join((os.path.dirname(os.path.realpath(__file__)), "include")),
]

import numpy as np

include_dirs = [
    np.get_include(),
    "\\".join((os.path.dirname(os.path.realpath(__file__)), "include")),
]

extra_compile_args=["-fopenmp" if sys.platform == 'linux' else "/openmp"]
setup(
    ext_modules=[
        Extension(
            "transforms84.transforms",
            sources=["include/transforms.c"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "transforms84.distances",
            sources=["include/distances.c"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "transforms84.helpers",
            sources=["include/helpers.c"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ],
)
