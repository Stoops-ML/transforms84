import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ECEF2ENUv

from .conftest import tol_double_atol, tol_float_atol


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
