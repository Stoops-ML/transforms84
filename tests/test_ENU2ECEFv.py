import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ENU2ECEFv


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ENU2ECEFv(tolerance_float_atol, dtype):
    rrm_local = DDM2RRM(np.array([[17.41], [78.27], [0]], dtype=dtype))
    uvw = np.array([[-27.6190], [-16.4298], [-0.3186]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2ECEFv(rrm_local, uvw),
            np.array([[27.9798], [-1.0993], [-15.7724]], dtype=np.float32),
            atol=tolerance_float_atol,
        )
    )
