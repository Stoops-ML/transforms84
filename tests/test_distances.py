import numpy as np
import pytest

from transforms84.distances import Haversine
from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84

# https://calculator.academy/haversine-distance-calculator/


def test_Haersine_raise_wrong_dtype():
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=np.float16)
    with pytest.raises(ValueError):
        Haversine(rrm_start, rrm_start, WGS84.mean_radius)  # type: ignore


def test_Haersine_raise_wrong_size():
    rrm_start = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [0], [1]], dtype=np.float32
    )
    with pytest.raises(ValueError):
        Haversine(rrm_start, rrm_start, WGS84.mean_radius)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_int(dtype):
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    assert np.isclose(Haversine(rrm_start, rrm_end, WGS84.mean_radius), 8120200.0)
    assert np.isclose(Haversine(rrm_end, rrm_start, WGS84.mean_radius), 8120200.0)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_with_height_int(dtype):
    rrm_start_with_height = np.array([[0], [0], [10]], dtype=dtype)
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius), 8120200.0
    )
    assert np.isclose(
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius),
        Haversine(rrm_start, rrm_end, WGS84.mean_radius),
    )
    assert np.isclose(Haversine(rrm_start, rrm_end, WGS84.mean_radius), 8120200.0)
    assert np.isclose(
        Haversine(rrm_end, rrm_start_with_height, WGS84.mean_radius), 8120200.0
    )
    assert np.isclose(Haversine(rrm_end, rrm_start, WGS84.mean_radius), 8120200.0)
    assert np.isclose(
        Haversine(rrm_start, rrm_end, WGS84.mean_radius),
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius),
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_one2many_int(dtype):
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
        Haversine(rrm_local, rrm_targets, WGS84.mean_radius)
        == Haversine(rrm_locals, rrm_targets, WGS84.mean_radius)
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine(dtype):
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(rrm_start, rrm_end, WGS84.mean_radius), 391225.574516907
    )
    assert np.isclose(
        Haversine(rrm_end, rrm_start, WGS84.mean_radius), 391225.574516907
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_with_height(dtype):
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius), 391225.574516907
    )
    assert np.isclose(
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius),
        Haversine(rrm_start, rrm_end, WGS84.mean_radius),
    )
    assert np.isclose(
        Haversine(rrm_start, rrm_end, WGS84.mean_radius), 391225.574516907
    )
    assert np.isclose(
        Haversine(rrm_end, rrm_start_with_height, WGS84.mean_radius), 391225.574516907
    )
    assert np.isclose(
        Haversine(rrm_end, rrm_start, WGS84.mean_radius), 391225.574516907
    )
    assert np.isclose(
        Haversine(rrm_start, rrm_end, WGS84.mean_radius),
        Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_one2many(dtype):
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
        Haversine(rrm_local, rrm_targets, WGS84.mean_radius)
        == Haversine(rrm_locals, rrm_targets, WGS84.mean_radius)
    )
