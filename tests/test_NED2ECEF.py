import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import NED2ECEF

from .conftest import tol_double_rtol, tol_float_rtol

# https://www.lddgo.net/en/coordinate/ecef-NED


def test_NED2ECEF_raise_wrong_dtype():
    XYZ = np.array([[1.3457e06], [-4.3509e06], [4.4523e06]], dtype=np.float16)
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)  # type: ignore


def test_NED2ECEF_raise_same_dtype():
    XYZ = np.array([[1.3457e06], [-4.3509e06], [4.4523e06]], dtype=np.float32)
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float64)
    with pytest.raises(ValueError):
        NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)


def test_NED2ECEF_raise_wrong_size():
    XYZ = np.array([[1.3457e06], [-4.3509e06], [4.4523e06]], dtype=np.float32)
    ref_point = np.array([[5010306], [2336344], [3170376.2], [1]], dtype=np.float64)
    with pytest.raises(ValueError):
        NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    with pytest.raises(ValueError):
        NED2ECEF(XYZ, ref_point, WGS84.a, WGS84.b)


@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_rtol), (np.int32, tol_float_rtol)]
)
def test_NED2ECEF_point_int(dtype, tol):
    XYZ = np.array([[-4350891], [1345660], [-4452314]], dtype=dtype)
    ref_point = np.array([[0], [0], [10]], dtype=dtype)
    out = NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 10830461, rtol=tol)
    assert np.isclose(out[1, 0], 1345660, rtol=tol)
    assert np.isclose(out[2, 0], -4350891, rtol=tol)


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_NED2ECEF_one2many_int(dtype):
    num_repeats = 3
    rrm_local = np.array([[[0], [0], [10]]], dtype=dtype)
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, num_repeats).T.reshape((-1, 3, 1))
    )
    ecef_target = np.array([[1345660], [-4350891], [4452314]], dtype=dtype)
    ecef_targets = np.ascontiguousarray(
        np.tile(ecef_target, num_repeats).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            NED2ECEF(
                rrm_locals,
                ecef_targets,
                WGS84.a,
                WGS84.b,
            ),
            NED2ECEF(
                rrm_local,
                ecef_targets,
                WGS84.a,
                WGS84.b,
            ),
        )
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_rtol), (np.float32, tol_float_rtol)]
)
def test_NED2ECEF_point(dtype, tol):
    XYZ = np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype)
    ref_point = DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype))
    out = NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 1.3457e06, rtol=tol)
    assert np.isclose(out[1, 0], -4.3509e06, rtol=tol)
    assert np.isclose(out[2, 0], 4.4523e06, rtol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_rtol), (np.float32, tol_float_rtol)]
)
def test_NED2ECEF_points(dtype, tol):
    XYZ = np.array(
        [
            [[1334.3], [-2544.4], [360.0]],
            [[1334.3], [-2544.4], [360.0]],
        ],
        dtype=dtype,
    )
    ref_point = DDM2RRM(
        np.array(
            [[[44.532], [-72.782], [1.699]], [[44.532], [-72.782], [1.699]]],
            dtype=dtype,
        )
    )
    out = NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 1.3457e06, rtol=tol))
    assert np.all(np.isclose(out[:, 1, 0], -4.3509e06, rtol=tol))
    assert np.all(np.isclose(out[:, 2, 0], 4.4523e06, rtol=tol))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_NED2ECEF_one2many_float(dtype):
    num_repeats = 3
    rrm_local = DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, num_repeats).T.reshape((-1, 3, 1))
    )
    ecef_target = np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype)
    ecef_targets = np.ascontiguousarray(
        np.tile(ecef_target, num_repeats).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            NED2ECEF(
                rrm_locals,
                ecef_targets,
                WGS84.a,
                WGS84.b,
            ),
            NED2ECEF(
                rrm_local,
                ecef_targets,
                WGS84.a,
                WGS84.b,
            ),
        )
    )
