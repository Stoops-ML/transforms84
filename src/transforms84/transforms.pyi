from typing import List, Set, Tuple, Union, overload

import numpy.typing as npt
import pandas as pd

from .definitions import SUPPORTED_TYPES

@overload
def geodetic2UTM(
    rrm_LLA: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to UTM coordinates.

    :param rrm_LLA: The geodetic coordinates as a NumPy array.
    :type rrm_LLA: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The UTM coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def geodetic2UTM(
    rad_lat: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to UTM coordinates.

    :param rad_lat: The latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon: The longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt: The altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The UTM coordinates as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def UTM2geodetic(
    mmUTM: npt.NDArray[SUPPORTED_TYPES],
    zone_number: int,
    zone_letter: str,
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert UTM coordinates to geodetic coordinates (latitude, longitude, altitude).

    :param mmUTM: The UTM coordinates as a NumPy array.
    :type mmUTM: npt.NDArray[SUPPORTED_TYPES]
    :param zone_number: The UTM zone number.
    :type zone_number: int
    :param zone_letter: The UTM zone letter.
    :type zone_letter: str
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The geodetic coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def UTM2geodetic(
    m_X: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    zone_number: int,
    zone_letter: str,
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert UTM coordinates to geodetic coordinates (latitude, longitude, altitude).

    :param m_X: The UTM easting coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y: The UTM northing coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param zone_number: The UTM zone number.
    :type zone_number: int
    :param zone_letter: The UTM zone letter.
    :type zone_letter: str
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The geodetic coordinates (latitude, longitude, altitude) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def geodetic2ECEF(
    rrm_LLA: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rrm_LLA: The geodetic coordinates as a NumPy array.
    :type rrm_LLA: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def geodetic2ECEF(
    rad_lat: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rad_lat: The latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon: The longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt: The altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates (X, Y, Z) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ECEF2geodetic(
    mmm_XYZ: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to geodetic coordinates (latitude, longitude, altitude).

    :param mmm_XYZ: The ECEF coordinates as a NumPy array.
    :type mmm_XYZ: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The geodetic coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ECEF2geodetic(
    m_X: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Z: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to geodetic coordinates (latitude, longitude, altitude).

    :param m_X: The ECEF X coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y: The ECEF Y coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Z: The ECEF Z coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Z: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The geodetic coordinates (latitude, longitude, altitude) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ECEF2ENU(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to ENU (East-North-Up) coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_target: The target ECEF coordinates as a NumPy array.
    :type mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ENU coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ECEF2ENU(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to ENU (East-North-Up) coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_X_target: The target ECEF X coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y_target: The target ECEF Y coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Z_target: The target ECEF Z coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ENU coordinates (East, North, Up) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ECEF2NED(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to NED (North-East-Down) coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_target: The target ECEF coordinates as a NumPy array.
    :type mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The NED coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ECEF2NED(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) coordinates to NED (North-East-Down) coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_X_target: The target ECEF X coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y_target: The target ECEF Y coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Z_target: The target ECEF Z coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The NED coordinates (North, East, Down) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ECEF2ENUv(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) velocity coordinates to ENU (East-North-Up) velocity coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_X_target: The target ECEF X velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y_target: The target ECEF Y velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Z_target: The target ECEF Z velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]

    :return: The ENU velocity coordinates (East, North, Up) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ECEF2ENUv(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES],
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) velocity coordinates to ENU (East-North-Up) velocity coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic velocity coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_target: The target ECEF velocity coordinates as a NumPy array.
    :type mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES]

    :return: The ENU velocity coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ECEF2NEDv(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES],
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) velocity coordinates to NED (North-East-Down) velocity coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic velocity coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_target: The target ECEF velocity coordinates as a NumPy array.
    :type mmm_XYZ_target: npt.NDArray[SUPPORTED_TYPES]

    :return: The NED velocity coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ECEF2NEDv(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) velocity coordinates to NED (North-East-Down) velocity coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_X_target: The target ECEF X velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_X_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Y_target: The target ECEF Y velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Y_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_Z_target: The target ECEF Z velocity coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_Z_target: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]

    :return: The NED velocity coordinates (North, East, Down) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ENU2ECEF(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ENU (East-North-Up) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_east: The East coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_north: The North coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_up: The Up coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates (X, Y, Z) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ENU2ECEF(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ENU (East-North-Up) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_local: The local ENU coordinates as a NumPy array.
    :type mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def NED2ECEF(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert NED (North-East-Down) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_north: The North coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_east: The East coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_down: The Down coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates (X, Y, Z) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def NED2ECEF(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES],
    m_semi_major_axis: float,
    m_semi_minor_axis: float,
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert NED (North-East-Down) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_local: The local NED coordinates as a NumPy array.
    :type mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES]
    :param m_semi_major_axis: The semi-major axis of the ellipsoid.
    :type m_semi_major_axis: float
    :param m_semi_minor_axis: The semi-minor axis of the ellipsoid.
    :type m_semi_minor_axis: float

    :return: The ECEF coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ENU2ECEFv(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ENU (East-North-Up) velocity coordinates to ECEF (Earth-Centered, Earth-Fixed) velocity coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_east: The East coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_north: The North coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_up: The Up coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]

    :return: The ECEF velocity coordinates (X, Y, Z) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ENU2ECEFv(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES],
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ENU (East-North-Up) velocity coordinates to ECEF (Earth-Centered, Earth-Fixed) velocity coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic velocity coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_local: The local ENU velocity coordinates as a NumPy array.
    :type mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES]

    :return: The ECEF velocity coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def NED2ECEFv(
    rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert NED (North-East-Down) velocity coordinates to ECEF (Earth-Centered, Earth-Fixed) velocity coordinates.

    :param rad_lat_local_origin: The local origin latitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lat_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param rad_lon_local_origin: The local origin longitude in radians as a NumPy array, pandas Series, or list of floats.
    :type rad_lon_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_alt_local_origin: The local origin altitude in meters as a NumPy array, pandas Series, or list of floats.
    :type m_alt_local_origin: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_north: The North velocity coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_east: The East velocity coordinate in meters as a NumPy array, pandas Series, or list of floats.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_down: The Down velocity coordinate in meters as a NumPy array, pandas
    :type m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series]

    :return: The ECEF velocity coordinates (X, Y, Z) as a tuple of NumPy arrays.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def NED2ECEFv(
    rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES],
    mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES],
) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert NED (North-East-Down) velocity coordinates to ECEF (Earth-Centered, Earth-Fixed) velocity coordinates.

    :param rrm_LLA_local_origin: The local origin geodetic velocity coordinates (latitude, longitude, altitude) as a NumPy array.
    :type rrm_LLA_local_origin: npt.NDArray[SUPPORTED_TYPES]
    :param mmm_XYZ_local: The local NED velocity coordinates as a NumPy array.
    :type mmm_XYZ_local: npt.NDArray[SUPPORTED_TYPES]

    :return: The ECEF velocity coordinates as a NumPy array.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def ENU2AER(
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert ENU (East-North-Up) coordinates to AER (Azimuth-Elevation-Range) coordinates.

    :param m_east: Array of East coordinates.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_north: Array of North coordinates.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_up: Array of Up coordinates.
    :type m_up: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]

    :returns: Tuple containing arrays of Azimuth, Elevation, and Range coordinates.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def ENU2AER(mmm_ENU: npt.NDArray[SUPPORTED_TYPES]) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert ENU (East-North-Up) coordinates to AER (Azimuth-Elevation-Range) coordinates.

    :param mmm_ENU: Array of ENU coordinates.
    :type mmm_ENU: npt.NDArray[SUPPORTED_TYPES]

    :returns: Array of AER coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def AER2ENU(
    rad_az: Union[
        npt.NDArray[SUPPORTED_TYPES],
        pd.Series,
        List[float],
        Set[float],
    ],
    rad_el: Union[
        npt.NDArray[SUPPORTED_TYPES],
        pd.Series,
        List[float],
        Set[float],
    ],
    m_range: Union[
        npt.NDArray[SUPPORTED_TYPES],
        pd.Series,
        List[float],
        Set[float],
    ],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert AER (Azimuth-Elevation-Range) coordinates to ENU (East-North-Up) coordinates.

    :param rad_az: Array of Azimuth angles in radians as a NumPy array, pandas Series, list, or set.
    :type rad_az: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float], Set[float]]
    :param rad_el: Array of Elevation angles in radians as a NumPy array, pandas Series, list, or set.
    :type rad_el: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float], Set[float]]
    :param m_range: Array of Range distances as a NumPy array, pandas Series, list, or set.
    :type m_range: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float], Set[float]]

    :returns: Tuple containing arrays of East, North, and Up coordinates.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES], npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def AER2ENU(rrm_AER: npt.NDArray[SUPPORTED_TYPES]) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert AER (Azimuth-Elevation-Range) coordinates to ENU (East-North-Up) coordinates.

    :param rrm_AER: Array of AER coordinates.
    :type rrm_AER: npt.NDArray[SUPPORTED_TYPES]

    :returns: Array of ENU coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def NED2AER(mmm_NED: npt.NDArray[SUPPORTED_TYPES]) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert NED (North-East-Down) coordinates to AER (Azimuth-Elevation-Range) coordinates.

    :param mmm_NED: Array of NED coordinates.
    :type mmm_NED: npt.NDArray[SUPPORTED_TYPES]

    :returns: Array of AER coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def NED2AER(
    m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
    m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert NED (North-East-Down) coordinates to AER (Azimuth-Elevation-Range) coordinates.

    :param m_north: Array of North coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_north: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_east: Array of East coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_east: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]
    :param m_down: Array of Down coordinates as a NumPy array, pandas Series, or list of floats.
    :type m_down: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series, List[float]]

    :returns: Tuple containing arrays of Azimuth, Elevation, and Range coordinates.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES]]
    """
    ...

@overload
def AER2NED(rrm_AER: npt.NDArray[SUPPORTED_TYPES]) -> npt.NDArray[SUPPORTED_TYPES]:
    """
    Convert AER (Azimuth-Elevation-Range) coordinates to NED (North-East-Down) coordinates.

    :param rrm_AER: Array of AER coordinates.
    :type rrm_AER: npt.NDArray[SUPPORTED_TYPES]

    :returns: Array of NED coordinates.
    :rtype: npt.NDArray[SUPPORTED_TYPES]
    """
    ...

@overload
def AER2NED(
    rad_az: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series],
    rad_el: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series],
    m_range: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series],
) -> Tuple[
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
    npt.NDArray[SUPPORTED_TYPES],
]:
    """
    Convert AER (Azimuth-Elevation-Range) coordinates to NED (North-East-Down) coordinates.

    :param rad_az: Array of Azimuth angles in radians as a NumPy array or pandas Series.
    :type rad_az: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series]
    :param rad_el: Array of Elevation angles in radians as a NumPy array or pandas Series.
    :type rad_el: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series]
    :param m_range: Array of Range distances as a NumPy array or pandas Series.
    :type m_range: Union[npt.NDArray[SUPPORTED_TYPES], pd.Series]

    :returns: Tuple containing arrays of North, East, and Down coordinates.
    :rtype: Tuple[npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES],
                npt.NDArray[SUPPORTED_TYPES]]
    """
    ...
