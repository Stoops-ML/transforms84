import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ENU2AER

from .conftest import float_type_pairs, tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_point_unrolled_list(dtype, tol):
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype)
    df = pd.DataFrame(ENU.T, columns=["E", "N", "U"])
    a, e, r = ENU2AER(df["E"].tolist(), df["N"].tolist(), df["U"].tolist())
    assert np.isclose(a, np.deg2rad(34.1160), atol=tol)
    assert np.isclose(e, np.deg2rad(4.1931), atol=tol)
    assert np.isclose(r, 15.1070, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_point_unrolled_pandas(dtype, tol):
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype)
    df = pd.DataFrame(ENU.T, columns=["E", "N", "U"])
    a, e, r = ENU2AER(df["E"], df["N"], df["U"])
    assert np.isclose(a, np.deg2rad(34.1160), atol=tol)
    assert np.isclose(e, np.deg2rad(4.1931), atol=tol)
    assert np.isclose(r, 15.1070, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_point_unrolled(dtype, tol):
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype)
    a, e, r = ENU2AER(ENU[0], ENU[1], ENU[2])
    assert np.isclose(a, np.deg2rad(34.1160), atol=tol)
    assert np.isclose(e, np.deg2rad(4.1931), atol=tol)
    assert np.isclose(r, 15.1070, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_point(dtype, tol):
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_points_unrolled(dtype, tol):
    ENU = np.array(
        [
            [[8.4504], [12.4737], [1.1046]],
            [[8.4504], [12.4737], [1.1046]],
        ],
        dtype=dtype,
    )
    a, e, r = ENU2AER(
        np.ascontiguousarray(ENU[:, 0, 0]),
        np.ascontiguousarray(ENU[:, 1, 0]),
        np.ascontiguousarray(ENU[:, 2, 0]),
    )
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_points_unrolled_list(dtype, tol):
    ENU = np.array(
        [
            [[8.4504], [12.4737], [1.1046]],
            [[8.4504], [12.4737], [1.1046]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    a, e, r = ENU2AER(df["E"].tolist(), df["N"].tolist(), df["U"].tolist())
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_points_unrolled_pandas(dtype, tol):
    ENU = np.array(
        [
            [[8.4504], [12.4737], [1.1046]],
            [[8.4504], [12.4737], [1.1046]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    a, e, r = ENU2AER(df["E"], df["N"], df["U"])
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_points(dtype, tol):
    ENU = np.array(
        [
            [[8.4504], [12.4737], [1.1046]],
            [[8.4504], [12.4737], [1.1046]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_parallel_unrolled(dtype, tol):
    ENU = np.ascontiguousarray(
        np.tile(np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    a, e, r = ENU2AER(
        np.ascontiguousarray(ENU[:, 0, 0]),
        np.ascontiguousarray(ENU[:, 1, 0]),
        np.ascontiguousarray(ENU[:, 2, 0]),
    )
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_parallel_unrolled_list(dtype, tol):
    ENU = np.ascontiguousarray(
        np.tile(np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    a, e, r = ENU2AER(df["E"].tolist(), df["N"].tolist(), df["U"].tolist())
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ENU2AER_parallel_unrolled_numbers_int(dtype_num):
    ENU = np.ascontiguousarray(
        np.tile(
            np.array([[8.4504], [12.4737], [1.1046]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    a, e, r = ENU2AER(dtype_num(df["E"]), dtype_num(df["N"]), dtype_num(df["U"]))
    a64, e64, r64 = ENU2AER(
        np.float64(dtype_num(df["E"])),
        np.float64(dtype_num(df["N"])),
        np.float64(dtype_num(df["U"])),
    )
    assert np.all(np.isclose(a, a64))
    assert np.all(np.isclose(e, e64))
    assert np.all(np.isclose(r, r64))


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_ENU2AER_parallel_unrolled_numbers_loop_int(dtype_num):
    ENU = np.ascontiguousarray(
        np.tile(
            np.array([[8.4504], [12.4737], [1.1046]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    for i_row in df.index:
        a, e, r = ENU2AER(
            dtype_num(df.loc[i_row, "E"]),
            dtype_num(df.loc[i_row, "N"]),
            dtype_num(df.loc[i_row, "U"]),
        )
        a64, e64, r64 = ENU2AER(
            np.float64(dtype_num(df["E"])),
            np.float64(dtype_num(df["N"])),
            np.float64(dtype_num(df["U"])),
        )
        assert np.all(np.isclose(a, a64))
        assert np.all(np.isclose(e, e64))
        assert np.all(np.isclose(r, r64))
        assert a.dtype == np.float64
        assert e.dtype == np.float64
        assert r.dtype == np.float64


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_ENU2AER_parallel_unrolled_numbers_loop(dtype_arr, dtype_num):
    ENU = np.ascontiguousarray(
        np.tile(
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype_arr), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    for i_row in df.index:
        a, e, r = ENU2AER(
            dtype_num(df.loc[i_row, "E"]),
            dtype_num(df.loc[i_row, "N"]),
            dtype_num(df.loc[i_row, "U"]),
        )
        assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol_float_atol))
        assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol_float_atol))
        assert np.all(np.isclose(r, 15.1070, atol=tol_float_atol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_parallel_unrolled_pandas(dtype, tol):
    ENU = np.ascontiguousarray(
        np.tile(np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    df = pd.DataFrame(
        {
            "E": ENU[:, 0, 0],
            "N": ENU[:, 1, 0],
            "U": ENU[:, 2, 0],
        }
    )
    a, e, r = ENU2AER(df["E"], df["N"], df["U"])
    assert np.all(np.isclose(a, np.deg2rad(34.1160), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(4.1931), atol=tol))
    assert np.all(np.isclose(r, 15.1070, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_parallel(dtype, tol):
    ENU = np.ascontiguousarray(
        np.tile(np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ENU2AER_point_int(dtype, tol):
    ENU = np.array([[1000], [100], [100]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ENU2AER_points_int(dtype, tol):
    ENU = np.array(
        [
            [[1000], [100], [100]],
            [[1000], [100], [100]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float64),
            atol=tol,
        ),
    )
