import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import ECEF2ENU, geodetic2ECEF

from .conftest import tol_double_atol, tol_float_atol

# https://www.lddgo.net/en/coordinate/ecef-enu


def test_ECEF2ENU_raise_wrong_dtype():
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float32
    )
    with pytest.raises(ValueError):
        ECEF2ENU(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float32)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2ENU(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2ENU(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore


def test_ECEF2ENU_raise_wrong_size():
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float32
    )
    ref_point = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        ECEF2ENU(ref_point, ENU, WGS84.a, WGS84.b)
    XYZ = np.array([[3906.67536618], [2732.16708], [1519.47079847]], dtype=np.float32)
    ref_point = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_point(dtype, tol):
    XYZ = np.array([[3906.67536618], [2732.16708], [1519.47079847]], dtype=dtype)
    ref_point = np.array([[0.1], [0.2], [5000]], dtype=dtype)
    out = ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 1901.5690521235, atol=tol)
    assert np.isclose(out[1, 0], 5316.9485968901, atol=tol)
    assert np.isclose(out[2, 0], -6378422.76482545, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_points(dtype, tol):
    XYZ = np.array(
        [
            [[3906.67536618], [2732.16708], [1519.47079847]],
            [[3906.67536618], [2732.16708], [1519.47079847]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[0.1], [0.2], [5000]], [[0.1], [0.2], [5000]]], dtype=dtype)
    out = ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 1901.5690521235, atol=tol))
    assert np.all(np.isclose(out[:, 1, 0], 5316.9485968901, atol=tol))
    assert np.all(np.isclose(out[:, 2, 0], -6378422.76482545, atol=tol))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_one2many(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    assert np.all(
        ECEF2ENU(
            rrm_locals, geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b), WGS84.a, WGS84.b
        )
        == ECEF2ENU(
            rrm_local, geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b), WGS84.a, WGS84.b
        )
    )
