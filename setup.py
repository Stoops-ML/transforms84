import os
import platform
import sys

import numpy as np
from setuptools import Extension, setup

if sys.platform.startswith("win"):
    if "MSC" in sys.version:
        ompcompileflags = ["-openmp"]
        omplinkflags = []
    else:
        # For non-MSVC toolchain e.g. gcc and clang with mingw
        ompcompileflags = ["-fopenmp"]
        omplinkflags = ["-fopenmp"]
elif sys.platform.startswith("darwin"):
    # This is a bit unusual but necessary...
    # llvm (clang) OpenMP is used for headers etc at compile time
    # Intel OpenMP (libiomp5) provides the link library.
    # They are binary compatible and may not safely coexist in a process, as
    # libiomp5 is more prevalent and often linked in for NumPy it is used
    # here!
    ompcompileflags = ["-fopenmp"]
    omplinkflags = ["-fopenmp=libiomp5"]
    omppath = ["lib", "clang", "*", "include", "omp.h"]
else:
    ompcompileflags = ["-fopenmp"]
    if platform.machine() == "ppc64le":  # from Numba # noqa: SIM108
        omplinkflags = ["-fopenmp"]
    else:
        omplinkflags = ["-fopenmp"]

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
