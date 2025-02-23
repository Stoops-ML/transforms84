import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import geodetic2UTM

# https://www.latlong.net/lat-long-utm.html


def test_raise_wrong_dtype():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float16)
    with pytest.raises(ValueError):
        geodetic2UTM(in_arr, WGS84.a, WGS84.b)  # type: ignore


def test_raise_wrong_dtype_unrolled():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float16)
    with pytest.raises(ValueError):
        geodetic2UTM(in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b)


def test_raise_wrong_size():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5], [10]], dtype=np.float64)
    with pytest.raises(ValueError):
        geodetic2UTM(in_arr, WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_point_int_unrolled(dtype):
    in_arr = DDM2RRM(np.array([[31.0], [35.0], [100]], dtype=dtype))
    out_x, out_y = geodetic2UTM(in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b)
    assert np.isclose(out_x, 690950.46)
    assert np.isclose(out_y, 3431318.84)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_point_int_unrolled_list(dtype):
    in_arr = DDM2RRM(np.array([[31.0], [35.0], [100]], dtype=dtype))
    df = pd.DataFrame(in_arr.T, columns=["lat", "lon", "alt"])
    out_x, out_y = geodetic2UTM(
        df["lat"].tolist(), df["lon"].tolist(), df["alt"].tolist(), WGS84.a, WGS84.b
    )
    assert np.isclose(out_x, 690950.46)
    assert np.isclose(out_y, 3431318.84)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_point_int_unrolled_pandas(dtype):
    in_arr = DDM2RRM(np.array([[31.0], [35.0], [100]], dtype=dtype))
    df = pd.DataFrame(in_arr.T, columns=["lat", "lon", "alt"])
    out_x, out_y = geodetic2UTM(df["lat"], df["lon"], df["alt"], WGS84.a, WGS84.b)
    assert np.isclose(out_x, 690950.46)
    assert np.isclose(out_y, 3431318.84)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_point_int(dtype):
    in_arr = DDM2RRM(np.array([[31.0], [35.0], [100]], dtype=dtype))
    out = geodetic2UTM(in_arr, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 690950.46)
    assert np.isclose(out[1, 0], 3431318.84)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_points_int(dtype):
    in_arr = DDM2RRM(
        np.array(
            [
                [[31.0], [35.0], [100]],
                [[31.0], [35.0], [100]],
            ],
            dtype=dtype,
        )
    )
    out_x, out_y = geodetic2UTM(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        np.ascontiguousarray(in_arr[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 690950.46))
    assert np.all(np.isclose(out_y, 3431318.84))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled_list(dtype):
    in_arr = DDM2RRM(np.array([[31.750000], [35.550000], [100]], dtype=dtype))
    df = pd.DataFrame(in_arr.T, columns=["lat", "lon", "alt"])
    out_x, out_y = geodetic2UTM(
        df["lat"].tolist(), df["lon"].tolist(), df["alt"].tolist(), WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled_pandas(dtype):
    in_arr = DDM2RRM(np.array([[31.750000], [35.550000], [100]], dtype=dtype))
    df = pd.DataFrame(in_arr.T, columns=["lat", "lon", "alt"])
    out_x, out_y = geodetic2UTM(df["lat"], df["lon"], df["alt"], WGS84.a, WGS84.b)
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled(dtype):
    in_arr = DDM2RRM(np.array([[31.750000], [35.550000], [100]], dtype=dtype))
    out_x, out_y = geodetic2UTM(in_arr[0], in_arr[1], in_arr[2], WGS84.a, WGS84.b)
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point(dtype):
    in_arr = np.array([[31.750000], [35.550000], [100]], dtype=dtype)
    out = geodetic2UTM(DDM2RRM(in_arr), WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], 741548.22))
    assert np.all(np.isclose(out[1, 0], 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled_list(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "lat": in_arr[:, 0, 0],
            "lon": in_arr[:, 1, 0],
            "alt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y = geodetic2UTM(
        df["lat"].tolist(), df["lon"].tolist(), df["alt"].tolist(), WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled_pandas(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "lat": in_arr[:, 0, 0],
            "lon": in_arr[:, 1, 0],
            "alt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y = geodetic2UTM(df["lat"], df["lon"], df["alt"], WGS84.a, WGS84.b)
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled(dtype):
    in_arr = np.array(
        [
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
            [[np.deg2rad(31.750000)], [np.deg2rad(35.550000)], [5]],
        ],
        dtype=dtype,
    )
    out_x, out_y = geodetic2UTM(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        np.ascontiguousarray(in_arr[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


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
def test_parallel_unrolled_list(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(31.75)], [np.deg2rad(35.55)], [5]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "lat": in_arr[:, 0, 0],
            "lon": in_arr[:, 1, 0],
            "alt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y = geodetic2UTM(
        df["lat"].tolist(), df["lon"].tolist(), df["alt"].tolist(), WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_unrolled_pandas(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(31.75)], [np.deg2rad(35.55)], [5]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "lat": in_arr[:, 0, 0],
            "lon": in_arr[:, 1, 0],
            "alt": in_arr[:, 2, 0],
        }
    )
    out_x, out_y = geodetic2UTM(df["lat"], df["lon"], df["alt"], WGS84.a, WGS84.b)
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_unrolled(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(31.75)], [np.deg2rad(35.55)], [5]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    out_x, out_y = geodetic2UTM(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        np.ascontiguousarray(in_arr[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 741548.22))
    assert np.all(np.isclose(out_y, 3515555.26))


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
