import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import NED2AER


def test_NED2AER_raise_wrong_dtype():
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2AER(NED)  # type: ignore


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_NED2AER_point(tolerance_float_atol, dtype):
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_NED2AER_points(tolerance_float_atol, dtype):
    NED = np.array(
        [
            [[-9.1013], [4.1617], [4.2812]],
            [[-9.1013], [4.1617], [4.2812]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
        ),
    )
