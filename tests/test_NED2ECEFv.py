import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import NED2ECEFv

from .conftest import float_type_pairs, tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_atol),
        (np.float32, np.float64, tol_double_atol),
    ],
)
def test_NED2ECEFv_different_dtypes(dtype0, dtype1, tol):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype0))
    uvw = np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype1)
    out = NED2ECEFv(rrm_local, uvw)
    assert out.dtype == np.float64
    assert np.all(
        np.isclose(
            out,
            np.array([[530.2445], [492.1283], [396.3459]]),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv_unrolled(dtype, tol):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype))
    uvw = np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype)
    x, y, z = NED2ECEFv(
        rrm_local[0], rrm_local[1], rrm_local[2], uvw[0], uvw[1], uvw[2]
    )
    assert np.isclose(x, 530.2445, atol=tol)
    assert np.isclose(y, 492.1283, atol=tol)
    assert np.isclose(z, 396.3459, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv(dtype, tol):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype))
    uvw = np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2ECEFv(rrm_local, uvw),
            np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv_parallel_unrolled(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    x, y, z = NED2ECEFv(
        np.ascontiguousarray(rrm_local[:, 0, 0]),
        np.ascontiguousarray(rrm_local[:, 1, 0]),
        np.ascontiguousarray(rrm_local[:, 2, 0]),
        np.ascontiguousarray(uvw[:, 0, 0]),
        np.ascontiguousarray(uvw[:, 1, 0]),
        np.ascontiguousarray(uvw[:, 2, 0]),
    )
    assert np.all(np.isclose(x, 530.2445, atol=tol))
    assert np.all(np.isclose(y, 492.1283, atol=tol))
    assert np.all(np.isclose(z, 396.3459, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv_parallel_unrolled_list(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "rrm_local_0": rrm_local[:, 0, 0],
            "rrm_local_1": rrm_local[:, 1, 0],
            "rrm_local_2": rrm_local[:, 2, 0],
            "uvw_0": uvw[:, 0, 0],
            "uvw_1": uvw[:, 1, 0],
            "uvw_2": uvw[:, 2, 0],
        }
    )
    x, y, z = NED2ECEFv(
        df["rrm_local_0"].tolist(),
        df["rrm_local_1"].tolist(),
        df["rrm_local_2"].tolist(),
        df["uvw_0"].tolist(),
        df["uvw_1"].tolist(),
        df["uvw_2"].tolist(),
    )
    assert np.all(np.isclose(x, 530.2445, atol=tol))
    assert np.all(np.isclose(y, 492.1283, atol=tol))
    assert np.all(np.isclose(z, 396.3459, atol=tol))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_NED2ECEFv_parallel_unrolled_numbers_int(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "rrm_local_0": rrm_local[:, 0, 0],
            "rrm_local_1": rrm_local[:, 1, 0],
            "rrm_local_2": rrm_local[:, 2, 0],
            "uvw_0": uvw[:, 0, 0],
            "uvw_1": uvw[:, 1, 0],
            "uvw_2": uvw[:, 2, 0],
        }
    )
    x, y, z = NED2ECEFv(
        dtype_num(df["rrm_local_0"]),
        dtype_num(df["rrm_local_1"]),
        dtype_num(df["rrm_local_2"]),
        dtype_num(df["uvw_0"]),
        dtype_num(df["uvw_1"]),
        dtype_num(df["uvw_2"]),
    )
    x64, y64, z64 = NED2ECEFv(
        np.float64(dtype_num(df["rrm_local_0"])),
        np.float64(dtype_num(df["rrm_local_1"])),
        np.float64(dtype_num(df["rrm_local_2"])),
        np.float64(dtype_num(df["uvw_0"])),
        np.float64(dtype_num(df["uvw_1"])),
        np.float64(dtype_num(df["uvw_2"])),
    )
    assert np.all(np.isclose(x, x64))
    assert np.all(np.isclose(y, y64))
    assert np.all(np.isclose(z, z64))
    assert x.dtype == np.float64
    assert y.dtype == np.float64
    assert z.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_NED2ECEFv_parallel_unrolled_numbers_int_loop(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "rrm_local_0": rrm_local[:, 0, 0],
            "rrm_local_1": rrm_local[:, 1, 0],
            "rrm_local_2": rrm_local[:, 2, 0],
            "uvw_0": uvw[:, 0, 0],
            "uvw_1": uvw[:, 1, 0],
            "uvw_2": uvw[:, 2, 0],
        }
    )
    for i_row in df.index:
        x, y, z = NED2ECEFv(
            dtype_num(df.loc[i_row, "rrm_local_0"]),
            dtype_num(df.loc[i_row, "rrm_local_1"]),
            dtype_num(df.loc[i_row, "rrm_local_2"]),
            dtype_num(df.loc[i_row, "uvw_0"]),
            dtype_num(df.loc[i_row, "uvw_1"]),
            dtype_num(df.loc[i_row, "uvw_2"]),
        )
        x64, y64, z64 = NED2ECEFv(
            np.float64(dtype_num(df.loc[i_row, "rrm_local_0"])),
            np.float64(dtype_num(df.loc[i_row, "rrm_local_1"])),
            np.float64(dtype_num(df.loc[i_row, "rrm_local_2"])),
            np.float64(dtype_num(df.loc[i_row, "uvw_0"])),
            np.float64(dtype_num(df.loc[i_row, "uvw_1"])),
            np.float64(dtype_num(df.loc[i_row, "uvw_2"])),
        )
        assert np.all(np.isclose(x, x64))
        assert np.all(np.isclose(y, y64))
        assert np.all(np.isclose(z, z64))
        assert x.dtype == np.float64
        assert y.dtype == np.float64
        assert z.dtype == np.float64


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_NED2ECEFv_parallel_unrolled_numbers_loop(dtype_arr, dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype_arr)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype_arr), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "rrm_local_0": rrm_local[:, 0, 0],
            "rrm_local_1": rrm_local[:, 1, 0],
            "rrm_local_2": rrm_local[:, 2, 0],
            "uvw_0": uvw[:, 0, 0],
            "uvw_1": uvw[:, 1, 0],
            "uvw_2": uvw[:, 2, 0],
        }
    )
    for i_row in df.index:
        x, y, z = NED2ECEFv(
            dtype_num(df.loc[i_row, "rrm_local_0"]),
            dtype_num(df.loc[i_row, "rrm_local_1"]),
            dtype_num(df.loc[i_row, "rrm_local_2"]),
            dtype_num(df.loc[i_row, "uvw_0"]),
            dtype_num(df.loc[i_row, "uvw_1"]),
            dtype_num(df.loc[i_row, "uvw_2"]),
        )
        assert np.all(np.isclose(x, 530.2445, atol=tol_float_atol))
        assert np.all(np.isclose(y, 492.1283, atol=tol_float_atol))
        assert np.all(np.isclose(z, 396.3459, atol=tol_float_atol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv_parallel_unrolled_pandas(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "rrm_local_0": rrm_local[:, 0, 0],
            "rrm_local_1": rrm_local[:, 1, 0],
            "rrm_local_2": rrm_local[:, 2, 0],
            "uvw_0": uvw[:, 0, 0],
            "uvw_1": uvw[:, 1, 0],
            "uvw_2": uvw[:, 2, 0],
        }
    )
    x, y, z = NED2ECEFv(
        df["rrm_local_0"],
        df["rrm_local_1"],
        df["rrm_local_2"],
        df["uvw_0"],
        df["uvw_1"],
        df["uvw_2"],
    )
    assert np.all(np.isclose(x, 530.2445, atol=tol))
    assert np.all(np.isclose(y, 492.1283, atol=tol))
    assert np.all(np.isclose(z, 396.3459, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2ECEFv_parallel(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            NED2ECEFv(rrm_local, uvw),
            np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype),
            atol=tol,
        )
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_NED2ECEFv_int(dtype, tol):
    rrm_local = np.array([[61], [30], [0]], dtype=dtype)
    uvw = np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2ECEFv(rrm_local, uvw),
            np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype),
            atol=tol,
        )
    )
