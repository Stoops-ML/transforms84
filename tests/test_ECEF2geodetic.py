import numpy as np
import pandas as pd
import pytest

from transforms84.systems import WGS84
from transforms84.transforms import ECEF2geodetic

from .conftest import float_type_pairs, tol_double_atol, tol_float_atol

# https://www.redcrab-software.com/en/calculator/Ecef-to-Geo-Coordinates


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
def test_ECEF2geodetic_point_int_unrolled_list(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": in_arr[0],
            "y": in_arr[1],
            "z": in_arr[2],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        df["x"].tolist(), df["y"].tolist(), df["z"].tolist(), WGS84.a, WGS84.b
    )
    assert np.isclose(rad_lat, np.deg2rad(29.9999981))
    assert np.isclose(rad_lon, np.deg2rad(24.99999946))
    assert np.isclose(m_alt, 4.89436751, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_point_int_unrolled_pandas(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": in_arr[0],
            "y": in_arr[1],
            "z": in_arr[2],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(df["x"], df["y"], df["z"], WGS84.a, WGS84.b)
    assert np.isclose(rad_lat, np.deg2rad(29.9999981))
    assert np.isclose(rad_lon, np.deg2rad(24.99999946))
    assert np.isclose(m_alt, 4.89436751, atol=tol)


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
def test_ECEF2geodetic_point_unrolled_list(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": in_arr[0],
            "y": in_arr[1],
            "z": in_arr[2],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        df["x"].tolist(), df["y"].tolist(), df["z"].tolist(), WGS84.a, WGS84.b
    )
    assert np.isclose(rad_lat, np.deg2rad(30))
    assert np.isclose(rad_lon, np.deg2rad(25))
    assert np.isclose(m_alt, 5, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_point_unrolled_pandas(dtype, tol):
    in_arr = np.array([[5010306], [2336344], [3170376.2]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": in_arr[0],
            "y": in_arr[1],
            "z": in_arr[2],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(df["x"], df["y"], df["z"], WGS84.a, WGS84.b)
    assert np.isclose(rad_lat, np.deg2rad(30))
    assert np.isclose(rad_lon, np.deg2rad(25))
    assert np.isclose(m_alt, 5, atol=tol)


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
def test_ECEF2geodetic_points_int_unrolled_pandas(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376]],
            [[5010306], [2336344], [3170376]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(df["x"], df["y"], df["z"], WGS84.a, WGS84.b)
    assert np.all(np.isclose(rad_lat, np.deg2rad(29.9999981)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(24.99999946)))
    assert np.all(np.isclose(m_alt, 4.89436751, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2geodetic_points_int_unrolled_list(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376]],
            [[5010306], [2336344], [3170376]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        df["x"].tolist(), df["y"].tolist(), df["z"].tolist(), WGS84.a, WGS84.b
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
def test_ECEF2geodetic_points_unrolled_list(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376.2]],
            [[5010306], [2336344], [3170376.2]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        df["x"].tolist(), df["y"].tolist(), df["z"].tolist(), WGS84.a, WGS84.b
    )
    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_points_unrolled_pandas(dtype, tol):
    in_arr = np.array(
        [
            [[5010306], [2336344], [3170376.2]],
            [[5010306], [2336344], [3170376.2]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(df["x"], df["y"], df["z"], WGS84.a, WGS84.b)
    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


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
def test_ECEF2geodetic_parallel_unrolled_list(dtype, tol):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        df["x"].tolist(), df["y"].tolist(), df["z"].tolist(), WGS84.a, WGS84.b
    )

    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2geodetic_parallel_unrolled_pandas(dtype, tol):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(df["x"], df["y"], df["z"], WGS84.a, WGS84.b)

    assert np.all(np.isclose(rad_lat, np.deg2rad(30)))
    assert np.all(np.isclose(rad_lon, np.deg2rad(25)))
    assert np.all(np.isclose(m_alt, 5, atol=tol))


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_ECEF2geodetic_parallel_unrolled_numbers_loop(dtype_arr, dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=dtype_arr), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    for i_row in df.index:
        rad_lat, rad_lon, m_alt = ECEF2geodetic(
            dtype_num(df.loc[i_row, "x"]),
            dtype_num(df.loc[i_row, "y"]),
            dtype_num(df.loc[i_row, "z"]),
            WGS84.a,
            WGS84.b,
        )
        assert np.isclose(rad_lat, np.deg2rad(30))
        assert np.isclose(rad_lon, np.deg2rad(25))
        assert np.isclose(m_alt, 5, atol=tol_float_atol)
        assert rad_lat.dtype == dtype_num or rad_lat.dtype == np.float64
        assert rad_lon.dtype == dtype_num or rad_lon.dtype == np.float64
        assert m_alt.dtype == dtype_num or m_alt.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [int, np.int32, np.int64])
def test_ECEF2geodetic_parallel_unrolled_numbers_loop_int(dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    for i_row in df.index:
        rad_lat, rad_lon, m_alt = ECEF2geodetic(
            dtype_num(df.loc[i_row, "x"]),
            dtype_num(df.loc[i_row, "y"]),
            dtype_num(df.loc[i_row, "z"]),
            WGS84.a,
            WGS84.b,
        )
        rad_lat64, rad_lon64, m_alt64 = ECEF2geodetic(
            np.float64(dtype_num(df.loc[i_row, "x"])),
            np.float64(dtype_num(df.loc[i_row, "y"])),
            np.float64(dtype_num(df.loc[i_row, "z"])),
            WGS84.a,
            WGS84.b,
        )
        assert np.isclose(rad_lat, rad_lat64)
        assert np.isclose(rad_lon, rad_lon64)
        assert np.isclose(m_alt, m_alt64)


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ECEF2geodetic_parallel_unrolled_numbers_int(dtype_num):
    in_arr = np.ascontiguousarray(
        np.tile(
            np.array([[5010306], [2336344], [3170376.2]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": in_arr[:, 0, 0],
            "y": in_arr[:, 1, 0],
            "z": in_arr[:, 2, 0],
        }
    )
    rad_lat, rad_lon, m_alt = ECEF2geodetic(
        dtype_num(df["x"]),
        dtype_num(df["y"]),
        dtype_num(df["z"]),
        WGS84.a,
        WGS84.b,
    )
    rad_lat64, rad_lon64, m_alt64 = ECEF2geodetic(
        np.float64(dtype_num(df["x"])),
        np.float64(dtype_num(df["y"])),
        np.float64(dtype_num(df["z"])),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(rad_lat, rad_lat64))
    assert np.all(np.isclose(rad_lon, rad_lon64))
    assert np.all(np.isclose(m_alt, m_alt64))


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
