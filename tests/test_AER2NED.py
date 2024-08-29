import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2NED


def test_AER2NED_raise_wrong_dtype():
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2NED(AER)  # type: ignore


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2NED_point(tolerance_float_atol, dtype):
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tolerance_float_atol,
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2NED_points(tolerance_float_atol, dtype):
    AER = np.array(
        [
            [[155.427], [-23.161], [10.885]],
            [[155.427], [-23.161], [10.885]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tolerance_float_atol,
        ),
    )
