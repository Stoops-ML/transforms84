import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2ENU

from .conftest import float_type_pairs


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_point_unrolled_pandas(dtype):
    AER = DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype))
    df = pd.DataFrame(
        {
            "azimuth": AER[0],
            "elevation": AER[1],
            "range": AER[2],
        }
    )
    e, n, u = AER2ENU(df["azimuth"], df["elevation"], df["range"])
    assert np.isclose(e, 8.4504)
    assert np.isclose(n, 12.4737)
    assert np.isclose(u, 1.1046)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_point_unrolled(dtype):
    AER = DDM2RRM(np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype))
    e, n, u = AER2ENU(AER[0], AER[1], AER[2])
    assert np.isclose(e, 8.4504)
    assert np.isclose(n, 12.4737)
    assert np.isclose(u, 1.1046)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_point(dtype):
    AER = np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_AER2ENU_point_int(dtype):
    AER = np.array([[0], [0], [0]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2ENU(AER),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points(dtype):
    AER = np.array(
        [
            [[34.1160], [4.1931], [15.1070]],
            [[34.1160], [4.1931], [15.1070]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_unrolled_pandas(dtype):
    AER = DDM2RRM(
        np.array(
            [
                [[34.1160], [4.1931], [15.1070]],
                [[34.1160], [4.1931], [15.1070]],
            ],
            dtype=dtype,
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    e, n, u = AER2ENU(df["azimuth"], df["elevation"], df["range"])
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_unrolled_list(dtype):
    AER = DDM2RRM(
        np.array(
            [
                [[34.1160], [4.1931], [15.1070]],
                [[34.1160], [4.1931], [15.1070]],
            ],
            dtype=dtype,
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    e, n, u = AER2ENU(
        df["azimuth"].tolist(), df["elevation"].tolist(), df["range"].tolist()
    )
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_unrolled(dtype):
    AER = DDM2RRM(
        np.array(
            [
                [[34.1160], [4.1931], [15.1070]],
                [[34.1160], [4.1931], [15.1070]],
            ],
            dtype=dtype,
        )
    )
    e, n, u = AER2ENU(
        np.ascontiguousarray(AER[:, 0, 0]),
        np.ascontiguousarray(AER[:, 1, 0]),
        np.ascontiguousarray(AER[:, 2, 0]),
    )
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_parellel_unrolled(dtype):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype), 1000
            ).T.reshape((-1, 3, 1))
        )
    )

    e, n, u = AER2ENU(
        np.ascontiguousarray(AER[:, 0, 0]),
        np.ascontiguousarray(AER[:, 1, 0]),
        np.ascontiguousarray(AER[:, 2, 0]),
    )
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


@pytest.mark.parametrize("dtype_arr,dtype_num", float_type_pairs)
def test_AER2ENU_points_parellel_unrolled_numbers(dtype_arr, dtype_num):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype_arr), 1000
            ).T.reshape((-1, 3, 1))
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    for i_row in df.index:
        e, n, u = AER2ENU(
            dtype_num(df.loc[i_row, "azimuth"]),
            dtype_num(df.loc[i_row, "elevation"]),
            dtype_num(df.loc[i_row, "range"]),
        )
        assert np.isclose(e, 8.4504)
        assert np.isclose(n, 12.4737)
        assert np.isclose(u, 1.1046)
        assert isinstance(e, dtype_num) or e.dtype == np.float64
        assert isinstance(n, dtype_num) or n.dtype == np.float64
        assert isinstance(u, dtype_num) or u.dtype == np.float64


@pytest.mark.parametrize("dtype_num", [np.int32, np.int64])
def test_AER2ENU_points_parellel_unrolled_numbers_int(dtype_num):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float64), 1000
            ).T.reshape((-1, 3, 1))
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    e, n, u = AER2ENU(
        dtype_num(df["azimuth"]),
        dtype_num(df["elevation"]),
        dtype_num(df["range"]),
    )
    e64, n64, u64 = AER2ENU(
        np.float64(dtype_num(df["azimuth"])),
        np.float64(dtype_num(df["elevation"])),
        np.float64(dtype_num(df["range"])),
    )
    assert np.all(np.isclose(e, e64))
    assert np.all(np.isclose(n, n64))
    assert np.all(np.isclose(u, u64))


@pytest.mark.parametrize("dtype_num", [int, np.int32, np.int64])
def test_AER2ENU_points_parellel_unrolled_numbers_int_loop(dtype_num):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float64), 1000
            ).T.reshape((-1, 3, 1))
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    for i_row in df.index:
        e, n, u = AER2ENU(
            dtype_num(df.loc[i_row, "azimuth"]),
            dtype_num(df.loc[i_row, "elevation"]),
            dtype_num(df.loc[i_row, "range"]),
        )
        e64, n64, u64 = AER2ENU(
            np.float64(dtype_num(df.loc[i_row, "azimuth"])),
            np.float64(dtype_num(df.loc[i_row, "elevation"])),
            np.float64(dtype_num(df.loc[i_row, "range"])),
        )
        assert np.isclose(e, e64)
        assert np.isclose(n, n64)
        assert np.isclose(u, u64)
        assert e.dtype == np.float64
        assert n.dtype == np.float64
        assert u.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_parellel_unrolled_list(dtype):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype), 1000
            ).T.reshape((-1, 3, 1))
        )
    )
    df = pd.DataFrame(
        {
            "azimuth": AER[:, 0, 0],
            "elevation": AER[:, 1, 0],
            "range": AER[:, 2, 0],
        }
    )
    e, n, u = AER2ENU(
        df["azimuth"].to_list(), df["elevation"].to_list(), df["range"].to_list()
    )
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_parellel(dtype):
    AER = np.ascontiguousarray(
        np.tile(
            np.array([[34.1160], [4.1931], [15.1070]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_AER2ENU_points_int(dtype):
    AER = np.array(
        [
            [[0], [0], [0]],
            [[0], [0], [0]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2ENU(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )
