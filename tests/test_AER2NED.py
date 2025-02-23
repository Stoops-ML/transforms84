import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import AER2NED

from .conftest import tol_double_atol, tol_float_atol


def test_AER2NED_raise_wrong_dtype():
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2NED(AER)  # type: ignore


def test_AER2NED_raise_wrong_dtype_unrolled():
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=np.float16)
    with pytest.raises(ValueError):
        AER2NED(AER[0], AER[1], AER[2])


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_point_unrolled_pandas(dtype, tol):
    AER = DDM2RRM(np.array([[155.427], [-23.161], [10.885]], dtype=dtype))
    df = pd.DataFrame(
        {
            "azimuth": AER[0],
            "elevation": AER[1],
            "range": AER[2],
        }
    )
    n, e, d = AER2NED(df["azimuth"], df["elevation"], df["range"])
    assert np.isclose(n, -9.1013, atol=tol)
    assert np.isclose(e, 4.1617, atol=tol)
    assert np.isclose(d, 4.2812, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_point_unrolled(dtype, tol):
    AER = DDM2RRM(np.array([[155.427], [-23.161], [10.885]], dtype=dtype))
    n, e, d = AER2NED(AER[0], AER[1], AER[2])
    assert np.isclose(n, -9.1013, atol=tol)
    assert np.isclose(e, 4.1617, atol=tol)
    assert np.isclose(d, 4.2812, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_point(dtype, tol):
    AER = np.array([[155.427], [-23.161], [10.885]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_AER2NED_point_int(dtype):
    AER = np.array([[0], [0], [0]], dtype=dtype)
    assert np.all(
        np.isclose(
            AER2NED(AER),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_points_unrolled_pandas(dtype, tol):
    AER = DDM2RRM(
        np.array(
            [
                [[155.427], [-23.161], [10.885]],
                [[155.427], [-23.161], [10.885]],
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
    n, e, d = AER2NED(df["azimuth"], df["elevation"], df["range"])
    assert np.all(np.isclose(n, -9.1013, atol=tol))
    assert np.all(np.isclose(e, 4.1617, atol=tol))
    assert np.all(np.isclose(d, 4.2812, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_points_unrolled(dtype, tol):
    AER = DDM2RRM(
        np.array(
            [
                [[155.427], [-23.161], [10.885]],
                [[155.427], [-23.161], [10.885]],
            ],
            dtype=dtype,
        )
    )
    n, e, d = AER2NED(
        np.ascontiguousarray(AER[:, 0, 0]),
        np.ascontiguousarray(AER[:, 1, 0]),
        np.ascontiguousarray(AER[:, 2, 0]),
    )
    assert np.all(np.isclose(n, -9.1013, atol=tol))
    assert np.all(np.isclose(e, 4.1617, atol=tol))
    assert np.all(np.isclose(d, 4.2812, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_points(dtype, tol):
    AER = np.array(
        [
            [[155.427], [-23.161], [10.885]],
            [[155.427], [-23.161], [10.885]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_paralllel_unrolled_pandas(dtype, tol):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[155.427], [-23.161], [10.885]], dtype=dtype), 1000
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
    n, e, d = AER2NED(df["azimuth"], df["elevation"], df["range"])
    assert np.all(np.isclose(n, -9.1013, atol=tol))
    assert np.all(np.isclose(e, 4.1617, atol=tol))
    assert np.all(np.isclose(d, 4.2812, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_paralllel_unrolled(dtype, tol):
    AER = DDM2RRM(
        np.ascontiguousarray(
            np.tile(
                np.array([[155.427], [-23.161], [10.885]], dtype=dtype), 1000
            ).T.reshape((-1, 3, 1))
        )
    )
    n, e, d = AER2NED(
        np.ascontiguousarray(AER[:, 0, 0]),
        np.ascontiguousarray(AER[:, 1, 0]),
        np.ascontiguousarray(AER[:, 2, 0]),
    )
    assert np.all(np.isclose(n, -9.1013, atol=tol))
    assert np.all(np.isclose(e, 4.1617, atol=tol))
    assert np.all(np.isclose(d, 4.2812, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_AER2NED_paralllel(dtype, tol):
    AER = np.ascontiguousarray(
        np.tile(
            np.array([[155.427], [-23.161], [10.885]], dtype=dtype), 1000
        ).T.reshape((-1, 3, 1))
    )
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_AER2NED_points_int(dtype):
    AER = np.array(
        [
            [[0], [0], [0]],
            [[0], [0], [0]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            AER2NED(DDM2RRM(AER)),
            np.array([[8.4504], [12.4737], [1.1046]], dtype=dtype),
        ),
    )
