from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

NumberLikePy = Union[float, int]
NumberLikeNpy = Union[np.integer, np.floating]
NumberLike = Union[NumberLikePy, NumberLikeNpy]
ArrayLike = Union[
    npt.NDArray[NumberLikeNpy], "pd.Series[Union[float | int]]", List[NumberLike]
]
