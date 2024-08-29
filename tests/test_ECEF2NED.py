import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import ECEF2NED, geodetic2ECEF

# https://www.lddgo.net/en/coordinate/ecef-enu


def test_ECEF2NED_raise_wrong_dtype():
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float32
    )
    with pytest.raises(ValueError):
        ECEF2NED(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float32)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2NED(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2NED(ref_point, ENU, WGS84.a, WGS84.b)  # type: ignore


def test_ECEF2NED_raise_wrong_size():
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float32
    )
    ref_point = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        ECEF2NED(ref_point, ENU, WGS84.a, WGS84.b)
    XYZ = np.array([[3906.67536618], [2732.16708], [1519.47079847]], dtype=np.float32)
    ref_point = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        ECEF2NED(ref_point, XYZ, WGS84.a, WGS84.b)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2NED_point(dtype):
    XYZ = np.array([[1345660], [-4350891], [4452314]], dtype=dtype)
    ref_point = DDM2RRM(np.array([[44.532], [-72.782], [1699.0]], dtype=dtype))
    out = ECEF2NED(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 1334.3, rtol=0.001)
    assert np.isclose(out[1, 0], -2544.4, rtol=0.001)
    assert np.isclose(out[2, 0], 360.0, rtol=0.001)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2NED_points(dtype):
    XYZ = np.array(
        [
            [[1345660], [-4350891], [4452314]],
            [[1345660], [-4350891], [4452314]],
        ],
        dtype=dtype,
    )
    ref_point = DDM2RRM(
        np.array(
            [[[44.532], [-72.782], [1699.0]], [[44.532], [-72.782], [1699.0]]],
            dtype=dtype,
        )
    )
    out = ECEF2NED(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 1334.3, rtol=0.001))
    assert np.all(np.isclose(out[:, 1, 0], -2544.4, rtol=0.001))
    assert np.all(np.isclose(out[:, 2, 0], 360.0, rtol=0.001))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2NED_one2many(dtype):
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
        ECEF2NED(
            rrm_locals, geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b), WGS84.a, WGS84.b
        )
        == ECEF2NED(
            rrm_local, geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b), WGS84.a, WGS84.b
        )
    )
