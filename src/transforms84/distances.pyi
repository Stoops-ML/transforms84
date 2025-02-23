from typing import List, Union, overload

import numpy.typing as npt
import pandas as pd

from .definitions import SUPPORTED_TYPES

@overload
def Haversine(
    rrmStart: npt.NDArray[SUPPORTED_TYPES],
    rrmEnd: npt.NDArray[SUPPORTED_TYPES],
    m_radius_sphere: float,
) -> Union[float, npt.NDArray[SUPPORTED_TYPES]]:
    """Calculate the Haversine distance between two points on a sphere.

    :param rrmStart: The starting point coordinates as a NumPy array.
    :type rrmStart: npt.NDArray[SUPPORTED_TYPES]
    :param rrmEnd: The ending point coordinates as a NumPy array.
    :type rrmEnd: npt.NDArray[SUPPORTED_TYPES]
    :param m_radius_sphere: The radius of the sphere.
    :type m_radius_sphere: float

    :return: The Haversine distance between the two points. The return type matches the type of the input coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def Haversine(
    rad_lat_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lat_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_radius_sphere: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Calculate the Haversine distance between two points on a sphere.

    :param rad_lat_start: The starting point latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_start: The starting point longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_start: The starting point altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_start: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lat_end: The ending point latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_end: The ending point longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_end: The ending point altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_end: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_radius_sphere: The radius of the sphere in meters.
    :type m_radius_sphere: float

    :return: The Haversine distance between the two points. The return type matches the type of the input coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...
