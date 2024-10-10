import numpy as np
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
