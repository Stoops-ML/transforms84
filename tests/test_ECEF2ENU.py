import numpy as np
import pandas as pd
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


def test_ECEF2ENU_raise_wrong_dtype_unrolled():
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float32
    )
    with pytest.raises(ValueError):
        ECEF2ENU(
            ref_point[0],
            ref_point[1],
            ref_point[2],
            ENU[0],
            ENU[1],
            ENU[2],
            WGS84.a,
            WGS84.b,
        )
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float32)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2ENU(
            ref_point[0],
            ref_point[1],
            ref_point[2],
            ENU[0],
            ENU[1],
            ENU[2],
            WGS84.a,
            WGS84.b,
        )
    ref_point = np.array([[5010306], [2336344], [3170376.2]], dtype=np.float16)
    ENU = np.array(
        [[3906.67536618], [2732.16708], [1519.47079847], [1]], dtype=np.float16
    )
    with pytest.raises(ValueError):
        ECEF2ENU(
            ref_point[0],
            ref_point[1],
            ref_point[2],
            ENU[0],
            ENU[1],
            ENU[2],
            WGS84.a,
            WGS84.b,
        )


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
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_point_int_unrolled_pandas(dtype, tol):
    XYZ = np.array([[0], [0], [0]], dtype=dtype)
    ref_point = np.array([[1], [1], [500]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLatOrigin": ref_point[0],
            "radLonOrigin": ref_point[1],
            "mAltOrigin": ref_point[2],
            "mETarget": XYZ[0],
            "mYTarget": XYZ[1],
            "mZTarget": XYZ[2],
        }
    )
    m_x, m_y, m_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 0, atol=tol)
    assert np.isclose(m_y, 19458.6147548328, atol=tol)
    assert np.isclose(m_z, -6363502.5003553545, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_point_int_unrolled(dtype, tol):
    XYZ = np.array([[0], [0], [0]], dtype=dtype)
    ref_point = np.array([[1], [1], [500]], dtype=dtype)
    m_x, m_y, m_z = ECEF2ENU(
        ref_point[0],
        ref_point[1],
        ref_point[2],
        XYZ[0],
        XYZ[1],
        XYZ[2],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 0, atol=tol)
    assert np.isclose(m_y, 19458.6147548328, atol=tol)
    assert np.isclose(m_z, -6363502.5003553545, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_point_int(dtype, tol):
    XYZ = np.array([[0], [0], [0]], dtype=dtype)
    ref_point = np.array([[1], [1], [500]], dtype=dtype)
    out = ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.isclose(out[0, 0], 0, atol=tol)
    assert np.isclose(out[1, 0], 19458.6147548328, atol=tol)
    assert np.isclose(out[2, 0], -6363502.5003553545, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_point_unrolled_pandas(dtype, tol):
    XYZ = np.array([[3906.67536618], [2732.16708], [1519.47079847]], dtype=dtype)
    ref_point = np.array([[0.1], [0.2], [5000]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLatOrigin": ref_point[0],
            "radLonOrigin": ref_point[1],
            "mAltOrigin": ref_point[2],
            "mETarget": XYZ[0],
            "mYTarget": XYZ[1],
            "mZTarget": XYZ[2],
        }
    )
    m_x, m_y, m_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 1901.5690521235, atol=tol)
    assert np.isclose(m_y, 5316.9485968901, atol=tol)
    assert np.isclose(m_z, -6378422.76482545, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_point_unrolled(dtype, tol):
    XYZ = np.array([[3906.67536618], [2732.16708], [1519.47079847]], dtype=dtype)
    ref_point = np.array([[0.1], [0.2], [5000]], dtype=dtype)
    m_x, m_y, m_z = ECEF2ENU(
        ref_point[0],
        ref_point[1],
        ref_point[2],
        XYZ[0],
        XYZ[1],
        XYZ[2],
        WGS84.a,
        WGS84.b,
    )
    assert np.isclose(m_x, 1901.5690521235, atol=tol)
    assert np.isclose(m_y, 5316.9485968901, atol=tol)
    assert np.isclose(m_z, -6378422.76482545, atol=tol)


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
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_points_int_unrolled(dtype, tol):
    XYZ = np.array(
        [
            [[0], [0], [0]],
            [[0], [0], [0]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[1], [1], [500]], [[1], [1], [500]]], dtype=dtype)
    m_x, m_y, m_z = ECEF2ENU(
        np.ascontiguousarray(ref_point[:, 0]),
        np.ascontiguousarray(ref_point[:, 1]),
        np.ascontiguousarray(ref_point[:, 2]),
        np.ascontiguousarray(XYZ[:, 0]),
        np.ascontiguousarray(XYZ[:, 1]),
        np.ascontiguousarray(XYZ[:, 2]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 0, atol=tol))
    assert np.all(np.isclose(m_y, 19458.6147548328, atol=tol))
    assert np.all(np.isclose(m_z, -6363502.5003553545, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_points_int_unrolled_pandas(dtype, tol):
    XYZ = np.array(
        [
            [[0], [0], [0]],
            [[0], [0], [0]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[1], [1], [500]], [[1], [1], [500]]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLatOrigin": ref_point[:, 0, 0],
            "radLonOrigin": ref_point[:, 1, 0],
            "mAltOrigin": ref_point[:, 2, 0],
            "mETarget": XYZ[:, 0, 0],
            "mYTarget": XYZ[:, 1, 0],
            "mZTarget": XYZ[:, 2, 0],
        }
    )
    m_x, m_y, m_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 0, atol=tol))
    assert np.all(np.isclose(m_y, 19458.6147548328, atol=tol))
    assert np.all(np.isclose(m_z, -6363502.5003553545, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (np.int16, tol_float_atol),
        (np.int32, tol_float_atol),
        (np.int64, tol_float_atol),
    ],
)
def test_ECEF2ENU_points_int(dtype, tol):
    XYZ = np.array(
        [
            [[0], [0], [0]],
            [[0], [0], [0]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[1], [1], [500]], [[1], [1], [500]]], dtype=dtype)
    out = ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)
    assert np.all(np.isclose(out[:, 0, 0], 0, atol=tol))
    assert np.all(np.isclose(out[:, 1, 0], 19458.6147548328, atol=tol))
    assert np.all(np.isclose(out[:, 2, 0], -6363502.5003553545, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_points_unrolled(dtype, tol):
    XYZ = np.array(
        [
            [[3906.67536618], [2732.16708], [1519.47079847]],
            [[3906.67536618], [2732.16708], [1519.47079847]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[0.1], [0.2], [5000]], [[0.1], [0.2], [5000]]], dtype=dtype)
    m_x, m_y, m_z = ECEF2ENU(
        np.ascontiguousarray(ref_point[:, 0]),
        np.ascontiguousarray(ref_point[:, 1]),
        np.ascontiguousarray(ref_point[:, 2]),
        np.ascontiguousarray(XYZ[:, 0]),
        np.ascontiguousarray(XYZ[:, 1]),
        np.ascontiguousarray(XYZ[:, 2]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 1901.5690521235, atol=tol))
    assert np.all(np.isclose(m_y, 5316.9485968901, atol=tol))
    assert np.all(np.isclose(m_z, -6378422.76482545, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_ECEF2ENU_points_unrolled_pandas(dtype, tol):
    XYZ = np.array(
        [
            [[3906.67536618], [2732.16708], [1519.47079847]],
            [[3906.67536618], [2732.16708], [1519.47079847]],
        ],
        dtype=dtype,
    )
    ref_point = np.array([[[0.1], [0.2], [5000]], [[0.1], [0.2], [5000]]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radLatOrigin": ref_point[:, 0, 0],
            "radLonOrigin": ref_point[:, 1, 0],
            "mAltOrigin": ref_point[:, 2, 0],
            "mETarget": XYZ[:, 0, 0],
            "mYTarget": XYZ[:, 1, 0],
            "mZTarget": XYZ[:, 2, 0],
        }
    )
    m_x, m_y, m_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_x, 1901.5690521235, atol=tol))
    assert np.all(np.isclose(m_y, 5316.9485968901, atol=tol))
    assert np.all(np.isclose(m_z, -6378422.76482545, atol=tol))


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


@pytest.mark.parametrize(
    "dtype0,dtype1,tol",
    [
        (np.float64, np.float32, tol_double_atol),
        (np.float32, np.float64, tol_double_atol),
    ],
)
def test_ECEF2ENU_different_dtypes(dtype0, dtype1, tol):
    XYZ = np.array(
        [
            [[3906.67536618], [2732.16708], [1519.47079847]],
            [[3906.67536618], [2732.16708], [1519.47079847]],
        ],
        dtype=dtype0,
    )
    ref_point = np.array([[[0.1], [0.2], [5000]], [[0.1], [0.2], [5000]]], dtype=dtype1)
    out = ECEF2ENU(ref_point, XYZ, WGS84.a, WGS84.b)
    assert out.dtype == np.float64
    assert np.all(np.isclose(out[:, 0, 0], 1901.5690521235, atol=tol))
    assert np.all(np.isclose(out[:, 1, 0], 5316.9485968901, atol=tol))
    assert np.all(np.isclose(out[:, 2, 0], -6378422.76482545, atol=tol))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_ECEF2ENU_one2many_int_unrolled(dtype):
    rrm_target = DDM2RRM(np.array([[0], [0], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[1], [1], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        np.ascontiguousarray(rrm_locals[:, 0, 0]),
        np.ascontiguousarray(rrm_locals[:, 1, 0]),
        np.ascontiguousarray(rrm_locals[:, 2, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.ascontiguousarray(rrm_local[0]),
        np.ascontiguousarray(rrm_local[1]),
        np.ascontiguousarray(rrm_local[2]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_ECEF2ENU_one2many_int_unrolled_pandas(dtype):
    rrm_target = DDM2RRM(np.array([[0], [0], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[1], [1], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.array([df.loc[0, "radLatOrigin"]]),
        np.array([df.loc[0, "radLonOrigin"]]),
        np.array([df.loc[0, "mAltOrigin"]]),
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_ECEF2ENU_one2many_int_unrolled_list(dtype):
    rrm_target = DDM2RRM(np.array([[0], [0], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[1], [1], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"].tolist(),
        df["radLonOrigin"].tolist(),
        df["mAltOrigin"].tolist(),
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.array([df.loc[0, "radLatOrigin"]]),
        np.array([df.loc[0, "radLonOrigin"]]),
        np.array([df.loc[0, "mAltOrigin"]]),
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_ECEF2ENU_one2many_int(dtype):
    rrm_target = DDM2RRM(np.array([[0], [0], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[1], [1], [0]], dtype=dtype))
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_one2many_unrolled_list(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"].tolist(),
        df["radLonOrigin"].tolist(),
        df["mAltOrigin"].tolist(),
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        [df.loc[0, "radLatOrigin"]],  # type: ignore
        [df.loc[0, "radLonOrigin"]],  # type: ignore
        [df.loc[0, "mAltOrigin"]],  # type: ignore
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_one2many_unrolled_pandas(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.array([df.loc[0, "radLatOrigin"]]),
        np.array([df.loc[0, "radLonOrigin"]]),
        np.array([df.loc[0, "mAltOrigin"]]),
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_one2many_unrolled(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 3
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.ascontiguousarray(rrm_locals[:, 0, 0]),
        np.ascontiguousarray(rrm_locals[:, 1, 0]),
        np.ascontiguousarray(rrm_locals[:, 2, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        rrm_local[0],
        rrm_local[1],
        rrm_local[2],
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_parllel_unrolled(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 1000
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.ascontiguousarray(rrm_locals[:, 0, 0]),
        np.ascontiguousarray(rrm_locals[:, 1, 0]),
        np.ascontiguousarray(rrm_locals[:, 2, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        rrm_local[0],
        rrm_local[1],
        rrm_local[2],
        np.ascontiguousarray(mmm_ECEF_traget[:, 0, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 1, 0]),
        np.ascontiguousarray(mmm_ECEF_traget[:, 2, 0]),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_parllel_unrolled_pandas(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 1000
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        np.array([df.loc[0, "radLatOrigin"]]),
        np.array([df.loc[0, "radLonOrigin"]]),
        np.array([df.loc[0, "mAltOrigin"]]),
        df["mETarget"],
        df["mYTarget"],
        df["mZTarget"],
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_parllel_unrolled_list(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 1000
    rrm_targets = np.ascontiguousarray(
        np.tile(rrm_target, num_repeats).T.reshape((-1, 3, 1))
    )
    rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=dtype))
    rrm_locals = np.ascontiguousarray(
        np.tile(rrm_local, rrm_targets.shape[0]).T.reshape((-1, 3, 1))
    )
    mmm_ECEF_traget = geodetic2ECEF(rrm_targets, WGS84.a, WGS84.b)
    df = pd.DataFrame(
        {
            "radLatOrigin": rrm_locals[:, 0, 0],
            "radLonOrigin": rrm_locals[:, 1, 0],
            "mAltOrigin": rrm_locals[:, 2, 0],
            "mETarget": mmm_ECEF_traget[:, 0, 0],
            "mYTarget": mmm_ECEF_traget[:, 1, 0],
            "mZTarget": mmm_ECEF_traget[:, 2, 0],
        }
    )
    m_ENU_x, m_ENU_y, m_ENU_z = ECEF2ENU(
        df["radLatOrigin"].tolist(),
        df["radLonOrigin"].tolist(),
        df["mAltOrigin"].tolist(),
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    m_ENU_x1, m_ENU_y1, m_ENU_z1 = ECEF2ENU(
        [df.loc[0, "radLatOrigin"]],  # type: ignore
        [df.loc[0, "radLonOrigin"]],  # type: ignore
        [df.loc[0, "mAltOrigin"]],  # type: ignore
        df["mETarget"].tolist(),
        df["mYTarget"].tolist(),
        df["mZTarget"].tolist(),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(m_ENU_x, m_ENU_x1))
    assert np.all(np.isclose(m_ENU_y, m_ENU_y1))
    assert np.all(np.isclose(m_ENU_z, m_ENU_z1))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ECEF2ENU_parllel(dtype):
    rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=dtype))
    num_repeats = 1000
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
