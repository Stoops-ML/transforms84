import numpy as np
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import ECEF2geodetic

from .conftest import tol_double_atol, tol_float_atol

# https://www.redcrab-software.com/en/calculator/Ecef-to-Geo-Coordinates


def test_ECEF2geodetic_raise_wrong_dtype():
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    with pytest.raises(ValueError):
        ECEF2geodetic(in_arr, WGS84.a, WGS84.b)  # type: ignore


def test_ECEF2geodetic_raise_wrong_dtype_unrolled():
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    with pytest.raises(ValueError):
        ECEF2geodetic(in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b)


def test_ECEF2geodetic_raise_wrong_size():
    in_arr = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        ECEF2geodetic(in_arr, WGS84.a, WGS84.b)


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_point_int(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376]], dtype=dtype)
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], np.deg2rad(29.9999981))
    assert np.isclose(out[1, 0], np.deg2rad(24.99999946))
    assert np.isclose(out[2, 0], 4.89436751, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_point_int_unrolled(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376]], dtype=dtype)
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b
    )
    assert np.isclose(rad_lat, np.deg2rad(29.9999981))
    assert np.isclose(rad_lon, np.deg2rad(24.99999946))
    assert np.isclose(m_alt, 4.89436751, atol=tol)


@pytest.mark.skip(
    reason="16 bit integer results in overflow error when creating numpy array"
)
@pytest.mark.parametrize("dtype,tol", [(np.int16, tol_float_atol)])
def test_ECEF2geodetic_point_int16(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376]], dtype=dtype)
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], np.deg2rad(29.9999981))
    assert np.isclose(out[1, 0], np.deg2rad(24.99999946))
    assert np.isclose(out[2, 0], 4.89436751, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_point_unrolled(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=dtype)
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b
    )
    assert np.isclose(rad_lat, np.deg2rad(30))
    assert np.isclose(rad_lon, np.deg2rad(25))
    assert np.isclose(m_alt, 5, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_point(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=dtype)
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], np.deg2rad(30))
    assert np.isclose(out[1, 0], np.deg2rad(25))
    assert np.isclose(out[2, 0], 5, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_points_int_unrolled(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376]],
            [[5010306], [2336344], [3170376]],
        ],
        dtype=dtype,
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        np.ascontiguousarray(in_arr[:, 0]),
        np.ascontiguousarray(in_arr[:, 1]),
        np.ascontiguousarray(in_arr[:, 2]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(29.9999981)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(24.99999946)))
    assert np.all(np.isclose(m_alt, 4.89436751, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_points_int(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376]],
            [[5010306], [2336344], [3170376]],
        ],
        dtype=dtype,
    )
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(29.9999981)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(24.99999946)))
    assert np.all(np.isclose(out[:, 2, 0], 4.89436751, atol=tol))


@pytest.mark.skip(
    reason="16 bit integer results in overflow error when creating numpy array"
)
@pytest.mark.parametrize("dtype,tol", [(np.int16, tol_float_atol)])
def test_ECEF2geodetic_points_int16(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376]],
            [[5010306], [2336344], [3170376]],
        ],
        dtype=dtype,
    )
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(29.9999981)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(24.99999946)))
    assert np.all(np.isclose(out[:, 2, 0], 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_points_unrolled(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376.2]],
            [[5010306], [2336344], [3170376.2]],
        ],
        dtype=dtype,
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        np.ascontiguousarray(in_arr[:, 0]),
        np.ascontiguousarray(in_arr[:, 1]),
        np.ascontiguousarray(in_arr[:, 2]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_points(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376.2]],
            [[5010306], [2336344], [3170376.2]],
        ],
        dtype=dtype,
    )
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(30)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(25)))
    assert np.all(np.isclose(out[:, 2, 0], 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_parallel_unrolled(dtype, tol):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        np.ascontiguousarray(in_arr[:, 0]),
        np.ascontiguousarray(in_arr[:, 1]),
        np.ascontiguousarray(in_arr[:, 2]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_parallel(dtype, tol):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    out = ECEF2geodetic(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(30)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(25)))
    assert np.all(np.isclose(out[:, 2, 0], 5, atol=tol))
