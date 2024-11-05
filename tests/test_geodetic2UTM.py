import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import geodetic2UTM

# https://www.latlong.net/lat-long-utm.html


def test_raise_wrong_dtype():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float16)
    with pytest.raises(ValueError):
        geodetic2UTM(in_arr, WGS84.a, WGS84.b)  # type: ignore


def test_raise_wrong_size():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5], [10]], dtype=np.float64)
    with pytest.raises(ValueError):
        geodetic2UTM(in_arr, WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_point_int(dtype):
    in_arr = DDM2RRM(np.array([[31.0], [35.0], [100]], dtype=dtype))
    out = geodetic2UTM(in_arr, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 690950.46)
    assert np.isclose(out[1, 0], 3431318.84)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_points_int(dtype):
    in_arr = np.array(
        [
            [[31.0], [35.0], [100]],
            [[31.0], [35.0], [100]],
        ],
        dtype=dtype,
    )
    out = geodetic2UTM(DDM2RRM(in_arr), WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 690950.46))
    assert np.all(np.isclose(out[:, 1, 0], 3431318.84))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point(dtype):
    in_arr = np.array([[31.750000], [35.550000], [100]], dtype=dtype)
    out = geodetic2UTM(DDM2RRM(in_arr), WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], 741548.22))
    assert np.all(np.isclose(out[1, 0], 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
        ],
        dtype=dtype,
    )
    out = geodetic2UTM(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 741548.22))
    assert np.all(np.isclose(out[:, 1, 0], 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(31.75)], [np.deg2rad(35.55)], [5]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    out = geodetic2UTM(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 741548.22))
    assert np.all(np.isclose(out[:, 1, 0], 3515555.26))
