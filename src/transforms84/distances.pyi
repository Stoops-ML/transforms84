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
]:
    """Calculate the Haversine distance between two points on a sphere.

    :param rrmStart: The starting point coordinates as a NumPy array.
    :type rrmStart: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param rrmEnd: The ending point coordinates as a NumPy array.
    :type rrmEnd: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param m_radius_sphere: The radius of the sphere.
    :type m_radius_sphere: float

    :return: The Haversine distance between the two points. The return type matches the type of the input coordinates.
    :rtype: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    """
    ...

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
) -> npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]:
    """
    Calculate the Haversine distance between two points on a sphere.

    :param rad_lat_start: The starting point latitude in radians as a NumPy array.
    :type rad_lat_start: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param rad_lon_start: The starting point longitude in radians as a NumPy array.
    :type rad_lon_start: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param m_alt_start: The starting point altitude in meters as a NumPy array.
    :type m_alt_start: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param rad_lat_end: The ending point latitude in radians as a NumPy array.
    :type rad_lat_end: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param rad_lon_end: The ending point longitude in radians as a NumPy array.
    :type rad_lon_end: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param m_alt_end: The ending point altitude in meters as a NumPy array.
    :type m_alt_end: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    :param m_radius_sphere: The radius of the sphere in meters.
    :type m_radius_sphere: float

    :return: The Haversine distance between the two points. The return type matches the type of the input coordinates.
    :rtype: npt.NDArray[Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]]
    """
    ...
