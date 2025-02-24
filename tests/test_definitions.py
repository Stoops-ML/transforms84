from typing import get_args

import numpy as np

from transforms84.definitions import NumberLikeNpy


def test_supported_types():
    _types = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    _args = get_args(NumberLikeNpy)
    for _t in _types:
        assert issubclass(_t, _args)
