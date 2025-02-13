from typing import Union, overload

import numpy as np
import numpy.typing as npt

@overload
def Haversine(
    rrmStart: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    rrmEnd: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    m_radius_sphere: float,
) -> Union[
    float,
    npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]],
]: ...
@overload
def Haversine(
    rad_lat_start: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    rad_lon_start: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    m_alt_start: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    rad_lat_end: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    rad_lon_end: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    m_alt_end: npt.NDArray[
        Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
    ],
    m_radius_sphere: float,
) -> Union[
    float,
    npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]],
]: ...
