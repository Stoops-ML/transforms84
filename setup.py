import sys
import numpy as np
from setuptools import Extension, setup

extra_compile_args=["-fopenmp" if sys.platform == 'linux' else "/openmp"]
setup(
    ext_modules=[
        Extension(
            "transforms84.transforms",
            sources=["include/transforms.c"],
            include_dirs=[np.get_include(), "include/"],
            extra_compile_args=extra_compile_args
        ),
        Extension(
            "transforms84.distances",
            sources=["include/distances.c"],
            include_dirs=[np.get_include(), "include/"],
            extra_compile_args=extra_compile_args
        ),
        Extension(
            "transforms84.helpers",
            sources=["include/helpers.c"],
            include_dirs=[np.get_include(), "include/"],
            extra_compile_args=extra_compile_args
        ),
    ],
)
