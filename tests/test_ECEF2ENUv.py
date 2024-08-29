import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ECEF2ENUv


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENUv(tolerance_float_atol, dtype):
    rrm_local = DDM2RRM(np.array([[17.4114], [78.2700], [0]], dtype=dtype))
    uvw = np.array([[27.9799], [-1.0990], [-15.7723]], dtype=dtype)
    assert np.all(
        np.isclose(
            ECEF2ENUv(rrm_local, uvw),
            np.array([[-27.6190], [-16.4298], [-0.3186]]),
            atol=tolerance_float_atol,
        )
    )
