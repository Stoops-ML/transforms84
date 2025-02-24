import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ENU2ECEFv

from .conftest import tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_atol),
        (np.float32, np.float64, tol_double_atol),
    ],
)
def test_ENU2ECEFv_different_dtypes(dtype0, dtype1, tol):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype0))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype1)
    out = ENU2ECEFv(rrm_local, uvw)
    assert out.dtype == np.float64
    assert np.all(
        np.isclose(
            out,
            np.array([[27.9798], [-1.0993], [-15.7724]], dtype=np.float32),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_unrolled_list(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": rrm_local[0],
            "y": rrm_local[1],
            "z": rrm_local[2],
            "u": uvw[0],
            "v": uvw[1],
            "w": uvw[2],
        }
    )
    x, y, z = ENU2ECEFv(
        df["x"].tolist(),
        df["y"].tolist(),
        df["z"].tolist(),
        df["u"].tolist(),
        df["v"].tolist(),
        df["w"].tolist(),
    )
    assert np.isclose(x, 27.9798, atol=tol)
    assert np.isclose(y, -1.0993, atol=tol)
    assert np.isclose(z, -15.7724, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_unrolled_pandas(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype)
    df = pd.DataFrame(
        {
            "x": rrm_local[0],
            "y": rrm_local[1],
            "z": rrm_local[2],
            "u": uvw[0],
            "v": uvw[1],
            "w": uvw[2],
        }
    )
    x, y, z = ENU2ECEFv(df["x"], df["y"], df["z"], df["u"], df["v"], df["w"])
    assert np.isclose(x, 27.9798, atol=tol)
    assert np.isclose(y, -1.0993, atol=tol)
    assert np.isclose(z, -15.7724, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_unrolled(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype)
    x, y, z = ENU2ECEFv(
        rrm_local[0], rrm_local[1], rrm_local[2], uvw[0], uvw[1], uvw[2]
    )
    assert np.isclose(x, 27.9798, atol=tol)
    assert np.isclose(y, -1.0993, atol=tol)
    assert np.isclose(z, -15.7724, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv(dtype, tol):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2ECEFv(rrm_local, uvw),
            np.array([[27.9798], [-1.0993], [-15.7724]], dtype=np.float32),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_parallel_unrolled_list(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": rrm_local[:, 0, 0],
            "y": rrm_local[:, 1, 0],
            "z": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    x, y, z = ENU2ECEFv(
        df["x"].tolist(),
        df["y"].tolist(),
        df["z"].tolist(),
        df["u"].tolist(),
        df["v"].tolist(),
        df["w"].tolist(),
    )
    assert np.all(np.isclose(x, 27.9798, atol=tol))
    assert np.all(np.isclose(y, -1.0993, atol=tol))
    assert np.all(np.isclose(z, -15.7724, atol=tol))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ENU2ECEFv_parallel_unrolled_numbers_int(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": rrm_local[:, 0, 0],
            "y": rrm_local[:, 1, 0],
            "z": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    x, y, z = ENU2ECEFv(
        dtype_num(df["x"]),
        dtype_num(df["y"]),
        dtype_num(df["z"]),
        dtype_num(df["u"]),
        dtype_num(df["v"]),
        dtype_num(df["w"]),
    )
    x64, y64, z64 = ENU2ECEFv(
        np.float64(dtype_num(df["x"])),
        np.float64(dtype_num(df["y"])),
        np.float64(dtype_num(df["z"])),
        np.float64(dtype_num(df["u"])),
        np.float64(dtype_num(df["v"])),
        np.float64(dtype_num(df["w"])),
    )
    assert np.all(np.isclose(x, x64))
    assert np.all(np.isclose(y, y64))
    assert np.all(np.isclose(z, z64))
    assert x.dtype == np.float64
    assert y.dtype == np.float64
    assert z.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ENU2ECEFv_parallel_unrolled_numbers_loop_int(dtype_num):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=np.float64)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": rrm_local[:, 0, 0],
            "y": rrm_local[:, 1, 0],
            "z": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    for i_row in df.index:
        x, y, z = ENU2ECEFv(
            dtype_num(df.loc[i_row, "x"]),
            dtype_num(df.loc[i_row, "y"]),
            dtype_num(df.loc[i_row, "z"]),
            dtype_num(df.loc[i_row, "u"]),
            dtype_num(df.loc[i_row, "v"]),
            dtype_num(df.loc[i_row, "w"]),
        )
        x64, y64, z64 = ENU2ECEFv(
            np.float64(dtype_num(df.loc[i_row, "x"])),
            np.float64(dtype_num(df.loc[i_row, "y"])),
            np.float64(dtype_num(df.loc[i_row, "z"])),
            np.float64(dtype_num(df.loc[i_row, "u"])),
            np.float64(dtype_num(df.loc[i_row, "v"])),
            np.float64(dtype_num(df.loc[i_row, "w"])),
        )
        assert np.all(np.isclose(x, x64))
        assert np.all(np.isclose(y, y64))
        assert np.all(np.isclose(z, z64))
        assert x.dtype == np.float64
        assert y.dtype == np.float64
        assert z.dtype == np.float64


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_parallel_unrolled_pandas(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "x": rrm_local[:, 0, 0],
            "y": rrm_local[:, 1, 0],
            "z": rrm_local[:, 2, 0],
            "u": uvw[:, 0, 0],
            "v": uvw[:, 1, 0],
            "w": uvw[:, 2, 0],
        }
    )
    x, y, z = ENU2ECEFv(df["x"], df["y"], df["z"], df["u"], df["v"], df["w"])
    assert np.all(np.isclose(x, 27.9798, atol=tol))
    assert np.all(np.isclose(y, -1.0993, atol=tol))
    assert np.all(np.isclose(z, -15.7724, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_parallel_unrolled(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    x, y, z = ENU2ECEFv(
        np.ascontiguousarray(rrm_local[:, 0, 0]),
        np.ascontiguousarray(rrm_local[:, 1, 0]),
        np.ascontiguousarray(rrm_local[:, 2, 0]),
        np.ascontiguousarray(uvw[:, 0, 0]),
        np.ascontiguousarray(uvw[:, 1, 0]),
        np.ascontiguousarray(uvw[:, 2, 0]),
    )
    assert np.all(np.isclose(x, 27.9798, atol=tol))
    assert np.all(np.isclose(y, -1.0993, atol=tol))
    assert np.all(np.isclose(z, -15.7724, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2ECEFv_parallel(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            ENU2ECEFv(rrm_local, uvw),
            np.array([[27.9798], [-1.0993], [-15.7724]], dtype=np.float32),
            atol=tol,
        )
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ENU2ECEFv_int(dtype, tol):
    rrm_local = np.array([[0], [0], [0]], dtype=dtype)
    uvw = np.array([[-27], [-16.4298], [-0.3186]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2ECEFv(rrm_local, uvw),
            np.array([[27.9798], [-1.0993], [-15.7724]], dtype=np.float32),
            atol=tol,
        )
    )
