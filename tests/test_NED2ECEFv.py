import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import NED2ECEFv


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_NED2ECEFv_float(dtype):
    rrm_local = DDM2RRM(np.array([[61.64], [30.70], [0]], dtype=dtype))
    uvw = np.array([[-434.0403], [152.4451], [-684.6964]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2ECEFv(rrm_local, uvw),
            np.array([[530.2445], [492.1283], [396.3459]], dtype=np.float32),
        )
    )
