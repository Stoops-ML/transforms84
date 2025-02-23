import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2ENU


def test_AER2ENU_raise_wrong_dtype_unrolled():
    AER = np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2ENU(AER[0], AER[1], AER[2])


def test_AER2ENU_raise_wrong_dtype():
    AER = np.array([[34.1160], [4.1931], [15.1070]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2ENU(AER)  # type: ignore


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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_AER2ENU_points_parellel_unrolled_pandas(dtype):
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
    e, n, u = AER2ENU(df["azimuth"], df["elevation"], df["range"])
    assert np.all(np.isclose(e, 8.4504))
    assert np.all(np.isclose(n, 12.4737))
    assert np.all(np.isclose(u, 1.1046))


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
