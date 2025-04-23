.. transforms84 documentation master file, created by
   sphinx-quickstart on Wed Apr 23 11:10:14 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

transforms84 documentation
==========================

Python library for geographic system transformations with additional helper functions.

This package focuses on:
1. Performance
2. Support for different number of inputs:
   * Ideal array shape of `(3,1)` and `(nPoints,3,1)` (as well as `(3,)` and `(nPoints,3)`)
   * Separate input for each axis in the coordinate system of size `(nPoints,)`
3. Support for different inputs types:
   * NumPy `ndarray`
   * Pandas `Series`
   * List
   * Float/int
4. Functions that adapt to differing input matrices shapes: one-to-one, many-to-many and one-to-many points.


.. toctree::
   :maxdepth: 2
   :caption: User Manual:

   user/index

.. toctree::
   :maxdepth: 2
   :caption: Reference Manual:

   reference/index

