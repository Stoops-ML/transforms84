import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import ENU2AER

from .conftest import tol_double_atol, tol_float_atol


def test_ENU2AER_raise_wrong_dtype():
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=np.float16)
    with pytest.raises(ValueError):
        ENU2AER(ENU)  # type: ignore


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_point(dtype, tol):
    ENU = np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype)
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float64)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ENU2AER_points(dtype, tol):
    ENU = np.array(
        [
            [[8.4504], [12.4737], [1.1046]],
            [[8.4504], [12.4737], [1.1046]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            ENU2AER(ENU),
            DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float64)),
            atol=tol,
        ),
    )
