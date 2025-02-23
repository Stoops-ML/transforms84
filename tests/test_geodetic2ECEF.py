import numpy as np
import pandas as pd
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import geodetic2ECEF


def test_geodetic2ECEF_raise_wrong_dtype():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float16)
    with pytest.raises(ValueError):
        geodetic2ECEF(in_arr, WGS84.a, WGS84.b)  # type: ignore


def test_geodetic2ECEF_raise_wrong_dtype_unrolled():
    in_arr = np.array([np.deg2rad(30)], dtype=np.float16)
    with pytest.raises(ValueError):
        geodetic2ECEF(in_arr, in_arr, in_arr, WGS84.a, WGS84.b)  # type: ignore


def test_geodetic2ECEF_raise_wrong_size():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5], [10]], dtype=np.float64)
    with pytest.raises(ValueError):
        geodetic2ECEF(in_arr, WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_point_int(dtype):
    in_arr = np.array([[1], [2], [5]], dtype=dtype)
    out = geodetic2ECEF(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], -1437504.9581578))
    assert np.all(np.isclose(out[1, 0], 3141005.63721087))
    assert np.all(np.isclose(out[2, 0], 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_point_int_unrolled_pandas(dtype):
    in_arr = np.array([[1], [2], [5]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_point_int_unrolled(dtype):
    in_arr = np.array([[1], [2], [5]], dtype=dtype)
    out_x, out_y, out_z = geodetic2ECEF(
        in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_points_int(dtype):
    in_arr = np.array(
        [
            [[1], [2], [5]],
            [[1], [2], [5]],
        ],
        dtype=dtype,
    )
    out = geodetic2ECEF(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], -1437504.9581578))
    assert np.all(np.isclose(out[:, 1, 0], 3141005.63721087))
    assert np.all(np.isclose(out[:, 2, 0], 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_points_int_unrolled_pandas(dtype):
    in_arr = np.array(
        [
            [[1], [2], [5]],
            [[1], [2], [5]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "radLat": in_arr[:, 0, 0],
            "radLon": in_arr[:, 1, 0],
            "mAlt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_geodetic2ECEF_points_int_unrolled(dtype):
    in_arr = np.array(
        [
            [[1], [2], [5]],
            [[1], [2], [5]],
        ],
        dtype=dtype,
    )
    out_x, out_y, out_z = geodetic2ECEF(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        np.ascontiguousarray(in_arr[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_point(dtype):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype)
    out = geodetic2ECEF(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], 5010302.11))
    assert np.all(np.isclose(out[1, 0], 2336342.24))
    assert np.all(np.isclose(out[2, 0], 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_point_unrolled_pandas(dtype):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_point_unrolled(dtype):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype)
    out_x, out_y, out_z = geodetic2ECEF(
        in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_points(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
        ],
        dtype=dtype,
    )
    out = geodetic2ECEF(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 5010302.11))
    assert np.all(np.isclose(out[:, 1, 0], 2336342.24))
    assert np.all(np.isclose(out[:, 2, 0], 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_points_unrolled_pandas(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "radLat": in_arr[:, 0, 0],
            "radLon": in_arr[:, 1, 0],
            "mAlt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_points_unrolled(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
            [[np.deg2rad(30)], [np.deg2rad(25)], [5]],
        ],
        dtype=dtype,
    )
    out_x, out_y, out_z = geodetic2ECEF(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        np.ascontiguousarray(in_arr[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_parallel(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    out = geodetic2ECEF(in_arr, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 5010302.11))
    assert np.all(np.isclose(out[:, 1, 0], 2336342.24))
    assert np.all(np.isclose(out[:, 2, 0], 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_parallel_unrolled_pandas(dtype):
    n = 1000
    df = pd.DataFrame(
        {
            "radLat": np.ascontiguousarray(np.repeat(np.deg2rad(30), n), dtype=dtype),
            "radLon": np.ascontiguousarray(np.repeat(np.deg2rad(25), n), dtype=dtype),
            "mAlt": np.ascontiguousarray(np.repeat(1, n), dtype=dtype),
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"], df["radLon"], df["mAlt"], WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_geodetic2ECEF_parallel_unrolled(dtype):
    n = 1000
    radLat = np.ascontiguousarray(np.repeat(np.deg2rad(30), n), dtype=dtype)
    radLon = np.ascontiguousarray(np.repeat(np.deg2rad(25), n), dtype=dtype)
    mAlt = np.ascontiguousarray(np.repeat(1, n), dtype=dtype)
    out_x, out_y, out_z = geodetic2ECEF(radLat, radLon, mAlt, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))
