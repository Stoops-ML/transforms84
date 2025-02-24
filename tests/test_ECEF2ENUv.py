import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ECEF2ENUv

from .conftest import float_type_pairs, tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype))
    uvw = np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype)
    assert np.all(
        np.isclose(
            ECEF2ENUv(rrm_local, uvw),
            np.array([[-27.6190], [-16.4298], [-0.3186]]),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_unrolled_pandas(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype))
    uvw = np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLat": rrm_local[0],
            "radLon": rrm_local[1],
            "mAlt": rrm_local[2],
            "u": uvw[0],
            "v": uvw[1],
            "w": uvw[2],
        }
    )
    e, n, u = ECEF2ENUv(
        df["radLat"], df["radLon"], df["mAlt"], df["u"], df["v"], df["w"]
    )
    assert np.isclose(e, [-27.6190], atol=tol)
    assert np.isclose(n, [-16.4298], atol=tol)
    assert np.isclose(u, [-0.3186], atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_unrolled(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype))
    uvw = np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype)
    e, n, u = ECEF2ENUv(
        rrm_local[0], rrm_local[1], rrm_local[2], uvw[0], uvw[1], uvw[2]
    )
    assert np.isclose(e, [-27.6190], atol=tol)
    assert np.isclose(n, [-16.4298], atol=tol)
    assert np.isclose(u, [-0.3186], atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_parallel(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            ECEF2ENUv(rrm_local, uvw),
            np.array([[-27.6190], [-16.4298], [-0.3186]]),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_parallel_unrolled_list(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radLat": rrm_local[:, 0, 0],
            "radLon": rrm_local[:, 1, 0],
            "mAlt": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    e, n, u = ECEF2ENUv(
        df["radLat"].tolist(),
        df["radLon"].tolist(),
        df["mAlt"].tolist(),
        df["u"].tolist(),
        df["v"].tolist(),
        df["w"].tolist(),
    )
    assert np.all(np.isclose(e, [-27.6190], atol=tol))
    assert np.all(np.isclose(n, [-16.4298], atol=tol))
    assert np.all(np.isclose(u, [-0.3186], atol=tol))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ECEF2ENUv_parallel_unrolled_numbers_int(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radLat": rrm_local[:, 0, 0],
            "radLon": rrm_local[:, 1, 0],
            "mAlt": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    e, n, u = ECEF2ENUv(
        dtype_num(df["radLat"]),
        dtype_num(df["radLon"]),
        dtype_num(df["mAlt"]),
        dtype_num(df["u"]),
        dtype_num(df["v"]),
        dtype_num(df["w"]),
    )
    e64, n64, u64 = ECEF2ENUv(
        np.float64(dtype_num(df["radLat"])),
        np.float64(dtype_num(df["radLon"])),
        np.float64(dtype_num(df["mAlt"])),
        np.float64(dtype_num(df["u"])),
        np.float64(dtype_num(df["v"])),
        np.float64(dtype_num(df["w"])),
    )
    assert np.all(np.isclose(e, e64))
    assert np.all(np.isclose(n, n64))
    assert np.all(np.isclose(u, u64))


@pytest.mark.parametrize("dtype_num", [int, np.int32, np.int64])
def test_ECEF2ENUv_parallel_unrolled_numbers_int_loop(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radLat": rrm_local[:, 0, 0],
            "radLon": rrm_local[:, 1, 0],
            "mAlt": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    for i_row in df.index:
        e, n, u = ECEF2ENUv(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            dtype_num(df.loc[i_row, "mAlt"]),
            dtype_num(df.loc[i_row, "u"]),
            dtype_num(df.loc[i_row, "v"]),
            dtype_num(df.loc[i_row, "w"]),
        )
        e64, n64, u64 = ECEF2ENUv(
            np.float64(dtype_num(df.loc[i_row, "radLat"])),
            np.float64(dtype_num(df.loc[i_row, "radLon"])),
            np.float64(dtype_num(df.loc[i_row, "mAlt"])),
            np.float64(dtype_num(df.loc[i_row, "u"])),
            np.float64(dtype_num(df.loc[i_row, "v"])),
            np.float64(dtype_num(df.loc[i_row, "w"])),
        )
        assert np.all(np.isclose(e, e64))
        assert np.all(np.isclose(n, n64))
        assert np.all(np.isclose(u, u64))
        assert e.dtype == np.float64
        assert n.dtype == np.float64
        assert u.dtype == np.float64


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_ECEF2ENUv_parallel_unrolled_numbers_loop(dtype_arr, dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype_arr)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype_arr), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radLat": rrm_local[:, 0, 0],
            "radLon": rrm_local[:, 1, 0],
            "mAlt": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    for i_row in df.index:
        e, n, u = ECEF2ENUv(
            dtype_num(df.loc[i_row, "radLat"]),
            dtype_num(df.loc[i_row, "radLon"]),
            dtype_num(df.loc[i_row, "mAlt"]),
            dtype_num(df.loc[i_row, "u"]),
            dtype_num(df.loc[i_row, "v"]),
            dtype_num(df.loc[i_row, "w"]),
        )
        assert np.all(np.isclose(e, [-27.6190], atol=tol_double_atol))
        assert np.all(np.isclose(n, [-16.4298], atol=tol_double_atol))
        assert np.all(np.isclose(u, [-0.3186], atol=tol_double_atol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_parallel_unrolled_pandas(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radLat": rrm_local[:, 0, 0],
            "radLon": rrm_local[:, 1, 0],
            "mAlt": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    e, n, u = ECEF2ENUv(
        df["radLat"], df["radLon"], df["mAlt"], df["u"], df["v"], df["w"]
    )
    assert np.all(np.isclose(e, [-27.6190], atol=tol))
    assert np.all(np.isclose(n, [-16.4298], atol=tol))
    assert np.all(np.isclose(u, [-0.3186], atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENUv_parallel_unrolled(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    e, n, u = ECEF2ENUv(
        np.ascontiguousarray(rrm_local[:, 0, 0]),
        np.ascontiguousarray(rrm_local[:, 1, 0]),
        np.ascontiguousarray(rrm_local[:, 2, 0]),
        np.ascontiguousarray(uvw[:, 0, 0]),
        np.ascontiguousarray(uvw[:, 1, 0]),
        np.ascontiguousarray(uvw[:, 2, 0]),
    )
    assert np.all(np.isclose(e, [-27.6190], atol=tol))
    assert np.all(np.isclose(n, [-16.4298], atol=tol))
    assert np.all(np.isclose(u, [-0.3186], atol=tol))


@pytest.mark.parametrize(
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_atol),
        (np.float32, np.float64, tol_double_atol),
    ],
)
def test_ECEF2ENUv_different_dtypes(dtype0, dtype1, tol):
    rrm_local = DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype0))
    uvw = np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype1)
    out = ECEF2ENUv(rrm_local, uvw)
    assert out.dtype == np.float64
    assert np.all(
        np.isclose(
            out,
            np.array([[-27.6190], [-16.4298], [-0.3186]]),
            atol=tol,
        )
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol",
    [
        (np.int64, tol_double_atol),
        (np.int32, tol_float_atol),
        (np.int16, tol_float_atol),
    ],
)
def test_ECEF2ENUv_int(dtype, tol):
    rrm_local = np.array([[0], [0], [0]], dtype=dtype)
    uvw = np.array([[1], [1], [1]], dtype=dtype)
    assert np.all(
        np.isclose(
            ECEF2ENUv(rrm_local, uvw),
            np.array([[-27.6190], [-16.4298], [-0.3186]]),
            atol=tol,
        )
    )
