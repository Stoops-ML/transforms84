import os
import platform
import sys

import numpy as np
from setuptools import Extension, setup

include_dirs = [
    np.get_include(),
    "\\".join((os.path.dirname(os.path.realpath(__file__)), "include")),
]

if sys.platform.startswith("win"):
    if "MSC" in sys.version:
        ompcompileflags = ["-openmp"]
        omplinkflags = []
    else:
        # For non-MSVC toolchain e.g. gcc and clang with mingw
        ompcompileflags = ["-fopenmp"]
        omplinkflags = ["-fopenmp"]
elif sys.platform.startswith("darwin"):
    if "clang" in os.popen("cc --version").read():
        # Disable OpenMP for macOS with clang
        ompcompileflags = []
        omplinkflags = []
    else:
        # llvm (clang) OpenMP is used for headers etc at compile time
        # Intel OpenMP (libiomp5) provides the link library.
        ompcompileflags = ["-fopenmp"]
        omplinkflags = ["-fopenmp=libiomp5"]
        omppath = ["lib", "clang", "*", "include", "omp.h"]
        include_dirs.append("/".join(omppath))  # Add omppath to include_dirs
else:
    ompcompileflags = ["-fopenmp"]
    if platform.machine() == "ppc64le":  # from Numba # noqa: SIM108
        omplinkflags = ["-fopenmp"]
    else:
        omplinkflags = ["-fopenmp"]

setup(
    ext_modules=[
        Extension(
            "transforms84.transforms",
            sources=["include/transforms.c"],
            include_dirs=include_dirs,
            extra_compile_args=ompcompileflags,
            extra_link_args=omplinkflags,
        ),
        Extension(
            "transforms84.distances",
            sources=["include/distances.c"],
            include_dirs=include_dirs,
            extra_compile_args=ompcompileflags,
            extra_link_args=omplinkflags,
        ),
        Extension(
            "transforms84.helpers",
            sources=["include/helpers.c"],
            include_dirs=include_dirs,
            extra_compile_args=ompcompileflags,
            extra_link_args=omplinkflags,
        ),
    ],
)
