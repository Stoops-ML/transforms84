from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from transforms84.distances import Haversine
from transforms84.helpers import DDM2RRM
from transforms84.systems import WGS84

# https://calculator.academy/haversine-distance-calculator/


def _test_Haversine_omp():
    N = 10000000
    print("generating N points")
    # rrm_start array of N point, from 33.0, 34.0, 0.0 to 33+i/N
    # rrm_start = np.array([
    #     [np.deg2rad(33 + i / N) for i in range(N)],
    #     [np.deg2rad(34 + i / N) for i in range(N)],
    #     [0 for i in range(N)],
    # ], dtype=np.float64)
    # rrm_end = np.array([
    #     [np.deg2rad(32 + i / N) for i in range(N)],
    #     [np.deg2rad(38 + i / N) for i in range(N)],
    #     [0 for i in range(N)],
    # ], dtype=np.float64)

    rrm_start = np.array(
        [
            np.deg2rad(np.linspace(33, 33 + 1 / N, N)),
            np.deg2rad(np.linspace(34, 34 + 1 / N, N)),
            np.zeros(N),
        ],
        dtype=np.float64,
    )

    rrm_end = np.array(
        [
            np.deg2rad(np.linspace(32, 32 + 1 / N, N)),
            np.deg2rad(np.linspace(38, 38 + 1 / N, N)),
            np.zeros(N),
        ],
        dtype=np.float64,
    )

    print(rrm_start.shape, rrm_end.shape)

    # generate [[]*N, []*N, []*N] array of points using numpy, with values from x=33.0,34.0,0.0 to X+i/N

    print("Haversine")
    perf_counter_start = perf_counter()
    Haversine(rrm_start, rrm_end, WGS84.mean_radius)
    perf_counter_end = perf_counter()
    print(f"Time: {perf_counter_end - perf_counter_start}")


# if __name__ == "__main__":
#     _test_Haversine_omp()


def test_Haersine_raise_wrong_size():
    rrm_start = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [0], [1]], dtype=np.float32
    )
    with pytest.raises(ValueError):
        Haversine(rrm_start, rrm_start, WGS84.mean_radius)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_int_unrolled(dtype):
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_int_unrolled_list(dtype):
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_int_unrolled_pandas(dtype):
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_int(dtype):
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    assert np.isclose(Haversine(rrm_start, rrm_end, WGS84.mean_radius), 8120200.0)
    assert np.isclose(Haversine(rrm_end, rrm_start, WGS84.mean_radius), 8120200.0)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_with_height_int_unrolled(dtype):
    rrm_start_with_height = np.array([[0], [0], [10]], dtype=dtype)
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_with_height_int_unrolled_pandas(dtype):
    rrm_start_with_height = np.array([[0], [0], [10]], dtype=dtype)
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
            "radStartWithHeightLat": rrm_start_with_height[0],
            "radStartWithHeightLon": rrm_start_with_height[1],
            "mStartWithHeightHeight": rrm_start_with_height[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16])
def test_Haversine_with_height_int_unrolled_list(dtype):
    rrm_start_with_height = np.array([[0], [0], [10]], dtype=dtype)
    rrm_start = np.array([[0], [0], [0]], dtype=dtype)
    rrm_end = np.array([[1], [1], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
            "radStartWithHeightLat": rrm_start_with_height[0],
            "radStartWithHeightLon": rrm_start_with_height[1],
            "mStartWithHeightHeight": rrm_start_with_height[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            WGS84.mean_radius,
        ),
        8120200.0,
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
    )


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
def test_Haversine_unrolled(dtype):
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_unrolled_list(dtype):
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_unrolled_pandas(dtype):
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
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
def test_Haversine_parallel_unrolled(dtype):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            Haversine(
                np.ascontiguousarray(rrm_start[:, 0, 0]),
                np.ascontiguousarray(rrm_start[:, 1, 0]),
                np.ascontiguousarray(rrm_start[:, 2, 0]),
                np.ascontiguousarray(rrm_end[:, 0, 0]),
                np.ascontiguousarray(rrm_end[:, 1, 0]),
                np.ascontiguousarray(rrm_end[:, 2, 0]),
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )
    assert np.all(
        np.isclose(
            Haversine(
                np.ascontiguousarray(rrm_end[:, 0, 0]),
                np.ascontiguousarray(rrm_end[:, 1, 0]),
                np.ascontiguousarray(rrm_end[:, 2, 0]),
                np.ascontiguousarray(rrm_start[:, 0, 0]),
                np.ascontiguousarray(rrm_start[:, 1, 0]),
                np.ascontiguousarray(rrm_start[:, 2, 0]),
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_parallel_unrolled_list(dtype):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[:, 0, 0],
            "radStartLon": rrm_start[:, 1, 0],
            "mStartHeight": rrm_start[:, 2, 0],
            "radEndLat": rrm_end[:, 0, 0],
            "radEndLon": rrm_end[:, 1, 0],
            "mEndHeight": rrm_end[:, 2, 0],
        }
    )
    assert np.all(
        np.isclose(
            Haversine(
                df["radStartLat"].tolist(),
                df["radStartLon"].tolist(),
                df["mStartHeight"].tolist(),
                df["radEndLat"].tolist(),
                df["radEndLon"].tolist(),
                df["mEndHeight"].tolist(),
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )
    assert np.all(
        np.isclose(
            Haversine(
                df["radEndLat"].tolist(),
                df["radEndLon"].tolist(),
                df["mEndHeight"].tolist(),
                df["radStartLat"].tolist(),
                df["radStartLon"].tolist(),
                df["mStartHeight"].tolist(),
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_parallel_unrolled_pandas(dtype):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[:, 0, 0],
            "radStartLon": rrm_start[:, 1, 0],
            "mStartHeight": rrm_start[:, 2, 0],
            "radEndLat": rrm_end[:, 0, 0],
            "radEndLon": rrm_end[:, 1, 0],
            "mEndHeight": rrm_end[:, 2, 0],
        }
    )
    assert np.all(
        np.isclose(
            Haversine(
                df["radStartLat"],
                df["radStartLon"],
                df["mStartHeight"],
                df["radEndLat"],
                df["radEndLon"],
                df["mEndHeight"],
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )
    assert np.all(
        np.isclose(
            Haversine(
                df["radEndLat"],
                df["radEndLon"],
                df["mEndHeight"],
                df["radStartLat"],
                df["radStartLon"],
                df["mStartHeight"],
                WGS84.mean_radius,
            ),
            391225.574516907,
        )
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, float])
def test_Haversine_parallel_unrolled_numbers(dtype):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[:, 0, 0],
            "radStartLon": rrm_start[:, 1, 0],
            "mStartHeight": rrm_start[:, 2, 0],
            "radEndLat": rrm_end[:, 0, 0],
            "radEndLon": rrm_end[:, 1, 0],
            "mEndHeight": rrm_end[:, 2, 0],
        }
    )
    for i_row in df.index:
        assert np.all(
            np.isclose(
                Haversine(
                    dtype(df.loc[i_row, "radStartLat"]),
                    dtype(df.loc[i_row, "radStartLon"]),
                    dtype(df.loc[i_row, "mStartHeight"]),
                    dtype(df.loc[i_row, "radEndLat"]),
                    dtype(df.loc[i_row, "radEndLon"]),
                    dtype(df.loc[i_row, "mEndHeight"]),
                    WGS84.mean_radius,
                ),
                391225.574516907,
            )
        )
        assert np.all(
            np.isclose(
                Haversine(
                    dtype(df.loc[i_row, "radEndLat"]),
                    dtype(df.loc[i_row, "radEndLon"]),
                    dtype(df.loc[i_row, "mEndHeight"]),
                    dtype(df.loc[i_row, "radStartLat"]),
                    dtype(df.loc[i_row, "radStartLon"]),
                    dtype(df.loc[i_row, "mStartHeight"]),
                    WGS84.mean_radius,
                ),
                391225.574516907,
            )
        )


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_Haversine_parallel_unrolled_numbers_int(dtype_num):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[:, 0, 0],
            "radStartLon": rrm_start[:, 1, 0],
            "mStartHeight": rrm_start[:, 2, 0],
            "radEndLat": rrm_end[:, 0, 0],
            "radEndLon": rrm_end[:, 1, 0],
            "mEndHeight": rrm_end[:, 2, 0],
        }
    )
    assert np.all(
        np.isclose(
            Haversine(
                np.float64(dtype_num(df["radStartLat"])),
                np.float64(dtype_num(df["radStartLon"])),
                np.float64(dtype_num(df["mStartHeight"])),
                np.float64(dtype_num(df["radEndLat"])),
                np.float64(dtype_num(df["radEndLon"])),
                np.float64(dtype_num(df["mEndHeight"])),
                WGS84.mean_radius,
            ),
            Haversine(
                dtype_num(df["radEndLat"]),
                dtype_num(df["radEndLon"]),
                dtype_num(df["mEndHeight"]),
                dtype_num(df["radStartLat"]),
                dtype_num(df["radStartLon"]),
                dtype_num(df["mStartHeight"]),
                WGS84.mean_radius,
            ),
        )
    )


@pytest.mark.parametrize("dtype_num", [int, np.int32, np.int64])
def test_Haversine_parallel_unrolled_numbers_int_loop(dtype_num):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=np.float64), 1000
        ).T.reshape((-1, 3, 1))
    )
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[:, 0, 0],
            "radStartLon": rrm_start[:, 1, 0],
            "mStartHeight": rrm_start[:, 2, 0],
            "radEndLat": rrm_end[:, 0, 0],
            "radEndLon": rrm_end[:, 1, 0],
            "mEndHeight": rrm_end[:, 2, 0],
        }
    )
    for i_row in df.index:
        assert np.all(
            np.isclose(
                Haversine(
                    np.float64(dtype_num(df.loc[i_row, "radStartLat"])),
                    np.float64(dtype_num(df.loc[i_row, "radStartLon"])),
                    np.float64(dtype_num(df.loc[i_row, "mStartHeight"])),
                    np.float64(dtype_num(df.loc[i_row, "radEndLat"])),
                    np.float64(dtype_num(df.loc[i_row, "radEndLon"])),
                    np.float64(dtype_num(df.loc[i_row, "mEndHeight"])),
                    WGS84.mean_radius,
                ),
                Haversine(
                    dtype_num(df.loc[i_row, "radEndLat"]),
                    dtype_num(df.loc[i_row, "radEndLon"]),
                    dtype_num(df.loc[i_row, "mEndHeight"]),
                    dtype_num(df.loc[i_row, "radStartLat"]),
                    dtype_num(df.loc[i_row, "radStartLon"]),
                    dtype_num(df.loc[i_row, "mStartHeight"]),
                    WGS84.mean_radius,
                ),
            )
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_parallel(dtype):
    rrm_start = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    rrm_end = np.ascontiguousarray(
        np.tile(
            np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(Haversine(rrm_start, rrm_end, WGS84.mean_radius), 391225.574516907)
    )
    assert np.all(
        np.isclose(Haversine(rrm_end, rrm_start, WGS84.mean_radius), 391225.574516907)
    )


@pytest.mark.parametrize(
    "dtype0,dtype1", [(np.float64, np.float32), (np.float32, np.float64)]
)
def test_Haversine_point_2D_different_dtypes(dtype0, dtype1):
    """output for 2D arrays end point should be a float"""
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype0
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype0)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype1)
    out0 = Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius)
    out1 = Haversine(rrm_start, rrm_end, WGS84.mean_radius)
    out2 = Haversine(rrm_end, rrm_start_with_height, WGS84.mean_radius)
    out3 = Haversine(rrm_end, rrm_start, WGS84.mean_radius)
    assert isinstance(out0, float)
    assert isinstance(out1, float)
    assert isinstance(out2, float)
    assert isinstance(out3, float)
    assert np.isclose(out0, 391225.574516907)
    assert np.isclose(out1, out0)
    assert np.isclose(out1, 391225.574516907)
    assert np.isclose(out2, 391225.574516907)
    assert np.isclose(out3, 391225.574516907)
    assert np.isclose(out1, out3)


@pytest.mark.parametrize(
    "dtype0,dtype1", [(np.float64, np.float32), (np.float32, np.float64)]
)
def test_Haversine_point_3D_end_point_different_dtypes(dtype0, dtype1):
    """output for a 3D array end point should be an array"""
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype0
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype0)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype1)
    out0 = Haversine(rrm_start_with_height, rrm_end[None, :], WGS84.mean_radius)
    out1 = Haversine(rrm_start, rrm_end[None, :], WGS84.mean_radius)
    out2 = Haversine(rrm_end, rrm_start_with_height[None, :], WGS84.mean_radius)
    out3 = Haversine(rrm_end, rrm_start[None, :], WGS84.mean_radius)
    assert isinstance(out0, np.ndarray)
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
    assert isinstance(out3, np.ndarray)
    assert out0.dtype == np.float64
    assert out1.dtype == np.float64
    assert out2.dtype == np.float64
    assert out3.dtype == np.float64
    assert np.isclose(out0, 391225.574516907)
    assert np.isclose(out1, out0)
    assert np.isclose(out1, 391225.574516907)
    assert np.isclose(out2, 391225.574516907)
    assert np.isclose(out3, 391225.574516907)
    assert np.isclose(out1, out3)


@pytest.mark.parametrize(
    "dtype0,dtype1", [(np.float64, np.float32), (np.float32, np.float64)]
)
def test_Haversine_points_different_dtypes(dtype0, dtype1):
    rrm_start_with_height = np.array(
        [
            [[np.deg2rad(33)], [np.deg2rad(34)], [100000]],
            [[np.deg2rad(33)], [np.deg2rad(34)], [100000]],
        ],
        dtype=dtype0,
    )
    rrm_start = np.array(
        [
            [[np.deg2rad(33)], [np.deg2rad(34)], [0]],
            [[np.deg2rad(33)], [np.deg2rad(34)], [0]],
        ],
        dtype=dtype0,
    )
    rrm_end = np.array(
        [
            [[np.deg2rad(32)], [np.deg2rad(38)], [0]],
            [[np.deg2rad(32)], [np.deg2rad(38)], [0]],
        ],
        dtype=dtype1,
    )
    out0 = Haversine(rrm_start_with_height, rrm_end, WGS84.mean_radius)
    out1 = Haversine(rrm_start, rrm_end, WGS84.mean_radius)
    out2 = Haversine(rrm_end, rrm_start_with_height, WGS84.mean_radius)
    out3 = Haversine(rrm_end, rrm_start, WGS84.mean_radius)
    assert isinstance(out0, np.ndarray)
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
    assert isinstance(out3, np.ndarray)
    assert out0.dtype == np.float64
    assert out1.dtype == np.float64
    assert out2.dtype == np.float64
    assert out3.dtype == np.float64
    assert np.all(np.isclose(out0, 391225.574516907))
    assert np.all(np.isclose(out1, out0))
    assert np.all(np.isclose(out1, 391225.574516907))
    assert np.all(np.isclose(out2, 391225.574516907))
    assert np.all(np.isclose(out3, 391225.574516907))
    assert np.all(np.isclose(out1, out3))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_with_height_unrolled(dtype):
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    assert np.isclose(
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            rrm_start[0],
            rrm_start[1],
            rrm_start[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
        Haversine(
            rrm_start_with_height[0],
            rrm_start_with_height[1],
            rrm_start_with_height[2],
            rrm_end[0],
            rrm_end[1],
            rrm_end[2],
            WGS84.mean_radius,
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_with_height_unrolled_list(dtype):
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
            "radStartWithHeightLat": rrm_start_with_height[0],
            "radStartWithHeightLon": rrm_start_with_height[1],
            "mStartWithHeightHeight": rrm_start_with_height[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"].tolist(),
            df["radStartLon"].tolist(),
            df["mStartHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartWithHeightLat"].tolist(),
            df["radStartWithHeightLon"].tolist(),
            df["mStartWithHeightHeight"].tolist(),
            df["radEndLat"].tolist(),
            df["radEndLon"].tolist(),
            df["mEndHeight"].tolist(),
            WGS84.mean_radius,
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_Haversine_with_height_unrolled_pandas(dtype):
    rrm_start_with_height = np.array(
        [[np.deg2rad(33)], [np.deg2rad(34)], [100000]], dtype=dtype
    )
    rrm_start = np.array([[np.deg2rad(33)], [np.deg2rad(34)], [0]], dtype=dtype)
    rrm_end = np.array([[np.deg2rad(32)], [np.deg2rad(38)], [0]], dtype=dtype)
    df = pd.DataFrame(
        {
            "radStartLat": rrm_start[0],
            "radStartLon": rrm_start[1],
            "mStartHeight": rrm_start[2],
            "radEndLat": rrm_end[0],
            "radEndLon": rrm_end[1],
            "mEndHeight": rrm_end[2],
            "radStartWithHeightLat": rrm_start_with_height[0],
            "radStartWithHeightLon": rrm_start_with_height[1],
            "mStartWithHeightHeight": rrm_start_with_height[2],
        }
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            WGS84.mean_radius,
        ),
        391225.574516907,
    )
    assert np.isclose(
        Haversine(
            df["radStartLat"],
            df["radStartLon"],
            df["mStartHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
        Haversine(
            df["radStartWithHeightLat"],
            df["radStartWithHeightLon"],
            df["mStartWithHeightHeight"],
            df["radEndLat"],
            df["radEndLon"],
            df["mEndHeight"],
            WGS84.mean_radius,
        ),
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
