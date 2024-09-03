import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ENU2ECEFv

from .conftest import tol_double_atol, tol_float_atol


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
