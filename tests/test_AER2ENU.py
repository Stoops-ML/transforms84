import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2ENU


def test_AER2ENU_raise_wrong_dtype():
    AER = np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2ENU(AER)  # type: ignore


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_point(dtype):
    AER = np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_float64_points(dtype):
    AER = np.array(
        [
            [[34.1160], [4.1931], [15.1070]],
            [[34.1160], [4.1931], [15.1070]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )
