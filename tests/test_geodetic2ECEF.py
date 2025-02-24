import numpy as np
import pandas as pd
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import geodetic2ECEF

from .conftest import float_type_pairs


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
def test_geodetic2ECEF_point_int_unrolled_list(dtype):
    in_arr = np.array([[1], [2], [5]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


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
def test_geodetic2ECEF_points_int_unrolled_list(dtype):
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
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, -1437504.9581578))
    assert np.all(np.isclose(out_y, 3141005.63721087))
    assert np.all(np.isclose(out_z, 5343772.65324303))


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
def test_geodetic2ECEF_point_unrolled_list(dtype):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


@pytest.mark.parametrize("dtype_num", [int, np.int32, np.int64])
def test_geodetic2ECEF_point_unrolled_numbers_int(dtype_num):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float64)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        dtype_num(df["radLat"]),
        dtype_num(df["radLon"]),
        dtype_num(df["mAlt"]),
        WGS84.a,
        WGS84.b,
    )
    out_x64, out_y64, out_z64 = geodetic2ECEF(
        np.float64(dtype_num(df["radLat"])),
        np.float64(dtype_num(df["radLon"])),
        np.float64(dtype_num(df["mAlt"])),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, out_x64))
    assert np.all(np.isclose(out_y, out_y64))
    assert np.all(np.isclose(out_z, out_z64))
    assert out_x.dtype == np.float64
    assert out_y.dtype == np.float64
    assert out_z.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_geodetic2ECEF_point_unrolled_numbers_loop_int(dtype_num):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=np.float64)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    for i_row in df.index:
        out_x, out_y, out_z = geodetic2ECEF(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            dtype_num(df.loc[i_row, "mAlt"]),
            WGS84.a,
            WGS84.b,
        )
        out_x64, out_y64, out_z64 = geodetic2ECEF(
            np.float64(dtype_num(df.loc[i_row, "radLat"])),
            np.float64(dtype_num(df.loc[i_row, "radLon"])),
            np.float64(dtype_num(df.loc[i_row, "mAlt"])),
            WGS84.a,
            WGS84.b,
        )
        assert np.all(np.isclose(out_x, out_x64))
        assert np.all(np.isclose(out_y, out_y64))
        assert np.all(np.isclose(out_z, out_z64))
        assert out_x.dtype == np.float64
        assert out_y.dtype == np.float64
        assert out_z.dtype == np.float64


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_geodetic2ECEF_point_unrolled_numbers_loop(dtype_arr, dtype_num):
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [5]], dtype=dtype_arr)
    df = pd.DataFrame(
        {
            "radLat": in_arr[0],
            "radLon": in_arr[1],
            "mAlt": in_arr[2],
        }
    )
    for i_row in df.index:
        out_x, out_y, out_z = geodetic2ECEF(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            dtype_num(df.loc[i_row, "mAlt"]),
            WGS84.a,
            WGS84.b,
        )
        assert np.all(np.isclose(out_x, 5010302.11))
        assert np.all(np.isclose(out_y, 2336342.24))
        assert np.all(np.isclose(out_z, 3170373.78))


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
def test_geodetic2ECEF_points_unrolled_list(dtype):
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
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


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
def test_geodetic2ECEF_parallel_unrolled_list(dtype):
    n = 1000
    df = pd.DataFrame(
        {
            "radLat": np.ascontiguousarray(np.repeat(np.deg2rad(30), n), dtype=dtype),
            "radLon": np.ascontiguousarray(np.repeat(np.deg2rad(25), n), dtype=dtype),
            "mAlt": np.ascontiguousarray(np.repeat(1, n), dtype=dtype),
        }
    )
    out_x, out_y, out_z = geodetic2ECEF(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out_x, 5010302.11))
    assert np.all(np.isclose(out_y, 2336342.24))
    assert np.all(np.isclose(out_z, 3170373.78))


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
