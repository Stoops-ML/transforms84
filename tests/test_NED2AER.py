import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import NED2AER

from .conftest import tol_double_atol, tol_float_atol


def test_NED2AER_raise_wrong_dtype():
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2AER(NED)  # type: ignore


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_NED2AER_point_int(dtype, tol):
    NED = np.array([[0], [1], [4]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2AER(NED),
            np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_NED2AER_points_int(dtype, tol):
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
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_point(dtype, tol):
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_points(dtype, tol):
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
            atol=tol,
        ),
    )
