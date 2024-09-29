import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import (
    AER2ENU,
    ECEF2ENU,
    ENU2AER,
    ENU2ECEF,
    ECEF2geodetic,
    geodetic2ECEF,
)

from .conftest import tol_double_atol, tol_float_atol


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_az_bounds(dtype, tol):
    """test that azimuth is between 0 and 359 degrees"""
    rrm_start = DDM2RRM(np.array([[32.06104], [35.10264], [0]], dtype=dtype))
    for i in range(360):
        rrm_direction = DDM2RRM(np.array([[i], [0], [28_000]], dtype=dtype))
        rrm_end = ECEF2geodetic(
            ENU2ECEF(rrm_start, AER2ENU(rrm_direction), WGS84.a, WGS84.b),
            WGS84.a,
            WGS84.b,
        )
        rrm_aer = ENU2AER(
            ECEF2ENU(
                rrm_start, geodetic2ECEF(rrm_end, WGS84.a, WGS84.b), WGS84.a, WGS84.b
            )
        )
        if not np.isclose(rrm_aer[0, 0], rrm_direction[0, 0], atol=tol):
            assert np.all(np.isclose(rrm_aer[1:], rrm_direction[1:], atol=tol)) and (
                np.isclose(rrm_aer[0, 0], rrm_direction[0, 0], atol=tol)
                or np.isclose(rrm_aer[0, 0] - 2 * np.pi, rrm_direction[0, 0], atol=tol)
            )
        else:
            assert np.all(np.isclose(rrm_aer[:2], rrm_direction[:2], atol=tol))
            assert np.isclose(rrm_aer[2], rrm_direction[2], atol=10)
        assert rrm_aer[0, 0] >= -0.00001 and rrm_aer[0, 0] <= (2 * np.pi) * 1.00001
        assert rrm_aer[1, 0] >= -np.pi and rrm_aer[1, 0] <= np.pi
        assert np.isclose(rrm_aer[1, 0], 0, atol=tol)
        assert rrm_aer.dtype == dtype


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_el_bounds(dtype, tol):
    """test that elevation is between -90 and 90 degrees"""
    rrm_start = DDM2RRM(np.array([[32.06104], [35.10264], [0]], dtype=dtype))
    for i in range(-89, 90):
        rrm_direction = DDM2RRM(np.array([[0], [i], [28_000]], dtype=dtype))
        rrm_end = ECEF2geodetic(
            ENU2ECEF(rrm_start, AER2ENU(rrm_direction), WGS84.a, WGS84.b),
            WGS84.a,
            WGS84.b,
        )
        rrm_aer = ENU2AER(
            ECEF2ENU(
                rrm_start, geodetic2ECEF(rrm_end, WGS84.a, WGS84.b), WGS84.a, WGS84.b
            )
        )
        if not np.isclose(rrm_aer[0, 0], rrm_direction[0, 0], atol=tol):
            assert np.all(np.isclose(rrm_aer[1], rrm_direction[1], atol=tol)) and (
                np.isclose(rrm_aer[0, 0], rrm_direction[0, 0], atol=tol)
                or np.isclose(rrm_aer[0, 0] - 2 * np.pi, rrm_direction[0, 0], atol=tol)
            )
        else:
            assert np.all(np.isclose(rrm_aer[:2], rrm_direction[:2], atol=tol))
        assert np.isclose(rrm_aer[2], rrm_direction[2], atol=10)
        assert rrm_aer[0, 0] >= -0.00001 and rrm_aer[0, 0] <= (2 * np.pi) * 1.00001
        assert rrm_aer[1, 0] >= -np.pi and rrm_aer[1, 0] <= np.pi
        assert rrm_aer.dtype == dtype
