import numpy as np
import pandas as pd
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import UTM2geodetic

from .conftest import float_type_pairs

# https://www.engineeringtoolbox.com/utm-latitude-longitude-d_1370.html


def test_raise_wrong_size():
    in_arr = np.array([[np.deg2rad(30)], [np.deg2rad(25)], [10]], dtype=np.float64)
    with pytest.raises(ValueError):
        UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_point_int_unrolled_list(dtype):
    in_arr = np.array([[690950.0], [3431318.0]], dtype=dtype)
    df = pd.DataFrame({"radLat": in_arr[0], "radLon": in_arr[1]})
    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_point_int_unrolled_pandas(dtype):
    in_arr = np.array([[690950.0], [3431318.0]], dtype=dtype)
    df = pd.DataFrame({"radLat": in_arr[0], "radLon": in_arr[1]})
    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"],
        df["radLon"],
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_point_int_unrolled(dtype):
    in_arr = np.array([[690950.0], [3431318.0]], dtype=dtype)
    rad_lat, rad_lon, m_alt = UTM2geodetic(
        in_arr[0], in_arr[1], 36, "R", WGS84.a, WGS84.b
    )
    assert np.isclose(rad_lat, np.deg2rad(30.999992))
    assert np.isclose(rad_lon, np.deg2rad(34.999995))
    assert np.isclose(m_alt, 0)


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_point_int(dtype):
    in_arr = np.array([[690950.0], [3431318.0]], dtype=dtype)
    out = UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], np.deg2rad(30.999992))
    assert np.isclose(out[1, 0], np.deg2rad(34.999995))
    assert np.isclose(out[2, 0], 0)


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_points_int(dtype):
    in_arr = np.array(
        [
            [[690950.0], [3431318.0]],
            [[690950.0], [3431318.0]],
        ],
        dtype=dtype,
    )
    out = UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(30.999992)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(34.999995)))
    assert np.all(np.isclose(out[:, 2, 0], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled_list(dtype):
    in_arr = np.array([[690950.46], [3431318.84]], dtype=dtype)
    df = pd.DataFrame({"radLat": in_arr[0], "radLon": in_arr[1]})
    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled_pandas(dtype):
    in_arr = np.array([[690950.46], [3431318.84]], dtype=dtype)
    df = pd.DataFrame({"radLat": in_arr[0], "radLon": in_arr[1]})
    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"],
        df["radLon"],
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_unrolled(dtype):
    in_arr = np.array([[690950.46], [3431318.84]], dtype=dtype)
    rad_lat, rad_lon, m_alt = UTM2geodetic(
        in_arr[0], in_arr[1], 36, "R", WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(31.0)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(35.0)))
    assert np.all(np.isclose(m_alt, 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point(dtype):
    in_arr = np.array([[690950.46], [3431318.84]], dtype=dtype)
    out = UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[1, 0], np.deg2rad(35.0)))
    assert np.all(np.isclose(out[2, 0], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled_list(dtype):
    in_arr = np.array(
        [
            [[690950.46], [3431318.84]],
            [[690950.46], [3431318.84]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})

    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled_pandas(dtype):
    in_arr = np.array(
        [
            [[690950.46], [3431318.84]],
            [[690950.46], [3431318.84]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})

    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"],
        df["radLon"],
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points_unrolled(dtype):
    in_arr = np.array(
        [
            [[690950.46], [3431318.84]],
            [[690950.46], [3431318.84]],
        ],
        dtype=dtype,
    )
    rad_lat, rad_lon, m_alt = UTM2geodetic(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(31.0)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(35.0)))
    assert np.all(np.isclose(m_alt, 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_points(dtype):
    in_arr = np.array(
        [
            [[690950.46], [3431318.84]],
            [[690950.46], [3431318.84]],
        ],
        dtype=dtype,
    )
    out = UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(35.0)))
    assert np.all(np.isclose(out[:, 2, 0], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_unrolled_list(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})

    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_parallel_unrolled_numbers_int(dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[690950.46], [3431318.84]], dtype=np.float64), 1000
        ).T.reshape((-1, 2, 1))
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})
    rad_lat, rad_lon, m_alt = UTM2geodetic(
        dtype_num(df["radLat"]),
        dtype_num(df["radLon"]),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    rad_lat64, rad_lon64, m_alt64 = UTM2geodetic(
        np.float64(dtype_num(df["radLat"])),
        np.float64(dtype_num(df["radLon"])),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, rad_lat64))
    assert np.all(np.isclose(rad_lon, rad_lon64))
    assert np.all(np.isclose(m_alt, m_alt64))
    assert rad_lat.dtype == np.float64
    assert rad_lon.dtype == np.float64
    assert m_alt.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_parallel_unrolled_numbers_int_loop(dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[690950.46], [3431318.84]], dtype=np.float64), 1000
        ).T.reshape((-1, 2, 1))
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})
    for i_row in df.index:
        rad_lat, rad_lon, m_alt = UTM2geodetic(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            36,
            "R",
            WGS84.a,
            WGS84.b,
        )
        rad_lat64, rad_lon64, m_alt64 = UTM2geodetic(
            np.float64(dtype_num(df.loc[i_row, "radLat"])),
            np.float64(dtype_num(df.loc[i_row, "radLon"])),
            36,
            "R",
            WGS84.a,
            WGS84.b,
        )
        assert np.all(np.isclose(rad_lat, rad_lat64))
        assert np.all(np.isclose(rad_lon, rad_lon64))
        assert np.all(np.isclose(m_alt, m_alt64))
        assert rad_lat.dtype == np.float64
        assert rad_lon.dtype == np.float64
        assert m_alt.dtype == np.float64


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_parallel_unrolled_numbers_loop(dtype_arr, dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype_arr), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})
    for i_row in df.index:
        rad_lat, rad_lon, m_alt = UTM2geodetic(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            36,
            "R",
            WGS84.a,
            WGS84.b,
        )
        assert np.all(np.isclose(rad_lat, np.deg2rad(31.0)))
        assert np.all(np.isclose(rad_lon, np.deg2rad(35.0)))
        assert np.all(np.isclose(m_alt, 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_unrolled_pandas(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    df = pd.DataFrame({"radLat": in_arr[:, 0, 0], "radLon": in_arr[:, 1, 0]})

    df["radLat"], df["radLon"], df["mAlt"] = UTM2geodetic(
        df["radLat"],
        df["radLon"],
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(df["radLat"], np.deg2rad(31.0)))
    assert np.all(np.isclose(df["radLon"], np.deg2rad(35.0)))
    assert np.all(np.isclose(df["mAlt"], 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel_unrolled(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    rad_lat, rad_lon, m_alt = UTM2geodetic(
        np.ascontiguousarray(in_arr[:, 0, 0]),
        np.ascontiguousarray(in_arr[:, 1, 0]),
        36,
        "R",
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(31.0)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(35.0)))
    assert np.all(np.isclose(m_alt, 0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parallel(dtype):
    in_arr = np.ascontiguousarray(
        np.tile(np.array([[690950.46], [3431318.84]], dtype=dtype), 1000).T.reshape(
            (-1, 2, 1)
        )
    )
    out = UTM2geodetic(in_arr, 36, "R", WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], np.deg2rad(31.0)))
    assert np.all(np.isclose(out[:, 1, 0], np.deg2rad(35.0)))
    assert np.all(np.isclose(out[:, 2, 0], 0))
