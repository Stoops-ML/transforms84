import numpy as np
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84
from transforms84.transforms import NED2ECEF

from .conftest import tol_double_rtol, tol_float_rtol

# https://www.lddgo.net/en/coordinate/ecef-NED


def test_NED2ECEF_raise_wrong_dtype_unrolled():
    XYZ = np.array([[1.3457e06], [-4.3509e06], [4.4523e06]], dtype=np.float16)
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2ECEF(
            ref_point[0],
            ref_point[1],
            ref_point[2],
            XYZ[0],
            XYZ[1],
            XYZ[2],
            WGS84.a,
            WGS84.b,
        )


def test_NED2ECEF_raise_wrong_dtype():
    XYZ = np.array([[1.3457e06], [-4.3509e06], [4.4523e06]], dtype=np.float16)
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)  # type: ignore


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
def test_NED2ECEF_point_int_unrolled(dtype, tol):
    XYZ = np.array([[-4350891], [1345660], [-4452314]], dtype=dtype)
    ref_point = np.array([[0], [0], [10]], dtype=dtype)
    m_x, m_y, m_z = NED2ECEF(
        ref_point[0],
        ref_point[1],
        ref_point[2],
        XYZ[0],
        XYZ[1],
        XYZ[2],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 10830461, rtol=tol)
    assert np.isclose(m_y, 1345660, rtol=tol)
    assert np.isclose(m_z, -4350891, rtol=tol)


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
def test_NED2ECEF_one2many_int_unrolled(dtype):
    num_repeats = 3
    rrm_local = np.array([[[0], [0], [10]]], dtype=dtype)
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, num_repeats).T.reshape((-1, 3, 1))
    )
    ecef_target = np.array([[1345660], [-4350891], [4452314]], dtype=dtype)
    ecef_targets = np.ascontiguousarray(
        np.tile(ecef_target, num_repeats).T.reshape((-1, 3, 1))
    )
    m_x, m_y, m_z = NED2ECEF(rrm_locals, ecef_targets, WGS84.a, WGS84.b)
    m_x1, m_y1, m_z1 = NED2ECEF(rrm_local, ecef_targets, WGS84.a, WGS84.b)
    assert np.all(np.isclose(m_x, m_x1))
    assert np.all(np.isclose(m_y, m_y1))
    assert np.all(np.isclose(m_z, m_z1))


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
def test_NED2ECEF_point_unrolled(dtype, tol):
    XYZ = np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype)
    ref_point = DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype))
    m_x, m_y, m_z = NED2ECEF(
        ref_point[0],
        ref_point[1],
        ref_point[2],
        XYZ[0],
        XYZ[1],
        XYZ[2],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 1.3457e06, rtol=tol)
    assert np.isclose(m_y, -4.3509e06, rtol=tol)
    assert np.isclose(m_z, 4.4523e06, rtol=tol)


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
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_rtol),
        (np.float32, np.float64, tol_double_rtol),
    ],
)
def test_NED2ECEF_different_dtypes(dtype0, dtype1, tol):
    XYZ = np.array(
        [
            [[1334.3], [-2544.4], [360.0]],
            [[1334.3], [-2544.4], [360.0]],
        ],
        dtype=dtype0,
    )
    ref_point = DDM2RRM(
        np.array(
            [[[44.532], [-72.782], [1.699]], [[44.532], [-72.782], [1.699]]],
            dtype=dtype1,
        )
    )
    out = NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    assert out.dtype == np.float64
    assert np.all(np.isclose(out[:, 0, 0], 1.3457e06, rtol=tol))
    assert np.all(np.isclose(out[:, 1, 0], -4.3509e06, rtol=tol))
    assert np.all(np.isclose(out[:, 2, 0], 4.4523e06, rtol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_rtol), (np.float32, tol_float_rtol)]
)
def test_NED2ECEF_points_unrolled(dtype, tol):
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
    m_x, m_y, m_z = NED2ECEF(
        np.ascontiguousarray(ref_point[:, 0, 0]),
        np.ascontiguousarray(ref_point[:, 1, 0]),
        np.ascontiguousarray(ref_point[:, 2, 0]),
        np.ascontiguousarray(XYZ[:, 0, 0]),
        np.ascontiguousarray(XYZ[:, 1, 0]),
        np.ascontiguousarray(XYZ[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 1.3457e06, rtol=tol))
    assert np.all(np.isclose(m_y, -4.3509e06, rtol=tol))
    assert np.all(np.isclose(m_z, 4.4523e06, rtol=tol))


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


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_rtol), (np.float32, tol_float_rtol)]
)
def test_NED2ECEF_parallel_unrolled(dtype, tol):
    XYZ = np.ascontiguousarray(
        np.tile(np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    ref_point = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    m_x, m_y, m_z = NED2ECEF(
        np.ascontiguousarray(ref_point[:, 0, 0]),
        np.ascontiguousarray(ref_point[:, 1, 0]),
        np.ascontiguousarray(ref_point[:, 2, 0]),
        np.ascontiguousarray(XYZ[:, 0, 0]),
        np.ascontiguousarray(XYZ[:, 1, 0]),
        np.ascontiguousarray(XYZ[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 1.3457e06, rtol=tol))
    assert np.all(np.isclose(m_y, -4.3509e06, rtol=tol))
    assert np.all(np.isclose(m_z, 4.4523e06, rtol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_rtol), (np.float32, tol_float_rtol)]
)
def test_NED2ECEF_parallel(dtype, tol):
    XYZ = np.ascontiguousarray(
        np.tile(np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    ref_point = np.ascontiguousarray(
        np.tile(
            DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype)), 1000
        ).T.reshape((-1, 3, 1))
    )
    out = NED2ECEF(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 1.3457e06, rtol=tol))
    assert np.all(np.isclose(out[:, 1, 0], -4.3509e06, rtol=tol))
    assert np.all(np.isclose(out[:, 2, 0], 4.4523e06, rtol=tol))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_NED2ECEF_one2many_float_unrolled(dtype):
    num_repeats = 3
    rrm_local = DDM2RRM(np.array([[44.532], [-72.782], [1.699]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, num_repeats).T.reshape((-1, 3, 1))
    )
    ecef_target = np.array([[1334.3], [-2544.4], [360.0]], dtype=dtype)
    ecef_targets = np.ascontiguousarray(
        np.tile(ecef_target, num_repeats).T.reshape((-1, 3, 1))
    )
    m_x, m_y, m_z = NED2ECEF(rrm_locals, ecef_targets, WGS84.a, WGS84.b)
    m_x1, m_y1, m_z1 = NED2ECEF(rrm_local, ecef_targets, WGS84.a, WGS84.b)
    assert np.all(np.isclose(m_x, m_x1))
    assert np.all(np.isclose(m_y, m_y1))
    assert np.all(np.isclose(m_z, m_z1))


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
