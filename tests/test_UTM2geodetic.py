import numpy as np
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import UTM2geodetic

# https://www.engineeringtoolbox.com/utm-latitude-longitude-d_1370.html


def test_raise_wrong_dtype():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)]], dtype=np.float16)
    with pytest.raises(ValueError):
        UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)  # type: ignore


def test_raise_wrong_size():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [10]], dtype=np.float64)
    with pytest.raises(ValueError):
        UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_point_int(dtype):
    in_arr = np.array([[690950.0], [3431318.0]], dtype=dtype)
    out = UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], np.deg2rad(30.999992))
    assert np.isclose(out[1, 0], np.deg2rad(34.999995))


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_points_int(dtype):
    in_arr = np.array(
        [
            [[690950.0], [3431318.0]],
            [[690950.0], [3431318.0]],
        ],
        dtype=dtype,
    )
    out = UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(30.999992)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(34.999995)))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point(dtype):
    in_arr = np.array([[690950.46], [3431318.84]], dtype=dtype)
    out = UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[1, 0], np.deg2rad(35.0)))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points(dtype):
    in_arr = np.array(
        [
            [[690950.46], [3431318.84]],
            [[690950.46], [3431318.84]],
        ],
        dtype=dtype,
    )
    out = UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(35.0)))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    out = UTM2geodetic(in_arr, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(35.0)))
