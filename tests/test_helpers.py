import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import (
    DDM2RRM,
    RRM2DDM,
    deg_angular_difference,
    rad_angular_difference,
    wrap,
)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wrap_array(dtype):
    for i in range(-180, 180):
        out = wrap(float(i) + np.arange(10, dtype=dtype), -90, 90)
        assert np.all(out >= -90)
        assert np.all(out <= 90)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wrap_arrays(dtype):
    for i in range(-180, 180):
        out = wrap(
            float(i) + np.arange(10, dtype=dtype),
            -90.0 * np.ones((10,), dtype=dtype),
            90.0 * np.ones((10,), dtype=dtype),
        )
        assert np.all(out >= -90)
        assert np.all(out <= 90)


@pytest.mark.parametrize("dtype", [float, int])
def test_wrap_no_array(dtype):
    for i in range(-180, 180):
        out = wrap(dtype(i), -90, 90)
        assert out >= -90
        assert out <= 90


def test_wrap_unequal_size():
    with pytest.raises(ValueError):
        wrap(np.array([-181.1, 100]), np.array([-180.0]), np.array([180.0]))


def test_wrap_bad_bounds():
    with pytest.raises(ValueError):
        wrap(np.array([-181.1, 100]), np.array([180.0]), np.array([-180.0]))
    with pytest.raises(ValueError):
        wrap(np.array([-181.1, 100]), 180.0, -180.0)
    with pytest.raises(ValueError):
        wrap(-181.1, 180.0, -180.0)


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int16, np.int32, np.int64]
)
def test_XXM2YYM_one_point(dtype):
    ddm_point = np.array([[30], [31], [0]], dtype=dtype)
    out = RRM2DDM(DDM2RRM(ddm_point))
    assert np.all(np.isclose(ddm_point, out))
    if np.issubdtype(dtype, np.integer):
        assert out.dtype == np.float64
    else:
        assert ddm_point.dtype == out.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_XXM2YYM_multiple_points(dtype):
    rrm_point = np.array([[[30], [31], [0]], [[30], [31], [0]]], dtype=dtype)
    out = DDM2RRM(RRM2DDM(rrm_point))
    assert np.all(np.isclose(rrm_point, out))
    assert rrm_point.dtype == out.dtype


def test_deg_angular_difference_smallest_angle1():
    for diff_v in range(0, 179):
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(diff == deg_angular_difference(diff, diff + diff_v, True))


def test_deg_angular_difference_largest_angle1():
    for diff_v in range(0, 1000):
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % 360.0 == deg_angular_difference(diff, diff + diff_v, False)
        )


def test_deg_angular_difference_largest_angle1_list():
    for diff_v in range(0, 1000):
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % 360.0 == deg_angular_difference(diff.tolist(), diff + diff_v, False)
        )
        assert np.all(
            diff % 360.0
            == deg_angular_difference(diff, (diff + diff_v).tolist(), False)
        )
        assert np.all(
            diff % 360.0
            == deg_angular_difference(diff.tolist(), (diff + diff_v).tolist(), False)
        )


def test_deg_angular_difference_largest_angle1_pandas():
    for diff_v in range(0, 1000):
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % 360.0
            == deg_angular_difference(pd.Series(diff), diff + diff_v, False)
        )
        assert np.all(
            diff % 360.0
            == deg_angular_difference(diff, pd.Series(diff + diff_v), False)
        )
        assert np.all(
            diff % 360.0
            == deg_angular_difference(pd.Series(diff), pd.Series(diff + diff_v), False)
        )


def test_deg_angular_difference_smallest_angle_floats():
    for diff in range(0, 179):
        for i in range(1000):
            assert diff == deg_angular_difference(
                float(diff * i), float(diff * (i + 1)), True
            )


def test_deg_angular_difference_largest_angle_floats():
    for diff in range(0, 1000):
        for i in range(1000):
            assert diff % 360.0 == deg_angular_difference(
                float(diff * i), float(diff * (i + 1)), False
            )


def test_deg_angular_difference_smallest_angle_ints():
    for diff in range(0, 179):
        for i in range(1000):
            assert diff == deg_angular_difference(diff * i, diff * (i + 1), True)


def test_deg_angular_difference_largest_angle_ints():
    for diff in range(0, 1000):
        for i in range(1000):
            assert diff % 360.0 == deg_angular_difference(
                diff * i, diff * (i + 1), False
            )


def test_rad_angular_difference_smallest_angle():
    for diff_v in range(0, 179):
        diff_v = np.deg2rad(diff_v)
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(diff == rad_angular_difference(diff, diff + diff_v, True))


def test_rad_angular_difference_largest_angle():
    for diff_v in range(0, 1000):
        diff_v = np.deg2rad(diff_v)
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % (2 * np.pi) == rad_angular_difference(diff, diff + diff_v, False)
        )


def test_rad_angular_difference_largest_angle_list():
    for diff_v in range(0, 1000):
        diff_v = np.deg2rad(diff_v)
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(diff.tolist(), diff + diff_v, False)
        )
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(diff, (diff + diff_v).tolist(), False)
        )
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(diff.tolist(), (diff + diff_v).tolist(), False)
        )


def test_rad_angular_difference_largest_angle_pandas():
    for diff_v in range(0, 1000):
        diff_v = np.deg2rad(diff_v)
        diff = np.ones((1000,), dtype=np.float32) * diff_v
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(pd.Series(diff), diff + diff_v, False)
        )
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(diff, pd.Series(diff + diff_v), False)
        )
        assert np.all(
            diff % (2 * np.pi)
            == rad_angular_difference(pd.Series(diff), pd.Series(diff + diff_v), False)
        )


def test_rad_angular_difference_smallest_angle_1():
    for diff in range(0, 179):
        for i in range(1000):
            diff_v = np.deg2rad(diff)
            assert np.isclose(
                diff_v, rad_angular_difference(diff_v * i, diff_v * (i + 1), True)
            )


def test_rad_angular_difference_largest_angle_1():
    for diff in range(0, 1000):
        for i in range(1000):
            diff_v = np.deg2rad(diff) % (2 * np.pi)
            assert np.isclose(
                diff_v % (2 * np.pi),
                rad_angular_difference(diff_v * i, diff_v * (i + 1), False),
            )
