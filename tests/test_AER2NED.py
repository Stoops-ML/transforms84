import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2NED

from .conftest import tol_double_atol, tol_float_atol


def test_AER2NED_raise_wrong_dtype():
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2NED(AER)  # type: ignore


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_point(dtype, tol):
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_points(dtype, tol):
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
            atol=tol,
        ),
    )
