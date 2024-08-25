"""
Because of the numba AOT compilation `pip install -e .` doesn't work. To run pytest:
1. run: `python setup.py bdist_wheel`
2. run: `pip uninstall CoorUtils`
3. run: `pip PATH/TO/COORUTILS_PACKAGE`
4. run: `pytest`
"""

import pytest


@pytest.fixture
def tolerance_float_atol() -> float:
    return 0.2


@pytest.fixture
def tolerance_double_atol() -> float:
    return 0.01
