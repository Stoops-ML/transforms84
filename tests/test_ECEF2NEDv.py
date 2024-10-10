import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ECEF2NEDv

from .conftest import tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2NEDv(dtype, tol):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype))
    uvw = np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype)
    assert np.all(
        np.isclose(
            ECEF2NEDv(rrm_local, uvw),
            np.array([[-434.0403], [152.4451], [-684.6964]]),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2NEDv_parallel(dtype, tol):
    rrm_local = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    uvw = np.ascontiguousarray(
        np.tile(
            np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            ECEF2NEDv(rrm_local, uvw),
            np.array([[-434.0403], [152.4451], [-684.6964]]),
            atol=tol,
        )
    )


@pytest.mark.parametrize(
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_atol),
        (np.float32, np.float64, tol_double_atol),
    ],
)
def test_ECEF2NEDv_different_dtypes(dtype0, dtype1, tol):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype0))
    uvw = np.array([[530.2445], [492.1283], [396.3459]], dtype=dtype1)
    out = ECEF2NEDv(rrm_local, uvw)
    assert out.dtype == np.float64
    assert np.all(
        np.isclose(
            out,
            np.array([[-434.0403], [152.4451], [-684.6964]]),
            atol=tol,
        )
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_ECEF2NEDv_int(dtype, tol):
    rrm_local = np.array([[0], [0], [0]], dtype=dtype)
    uvw = np.array([[10], [100], [400]], dtype=dtype)
    assert np.all(
        np.isclose(
            ECEF2NEDv(rrm_local, uvw),
            np.array([[-434.0403], [152.4451], [-684.6964]]),
            atol=tol,
        )
    )
