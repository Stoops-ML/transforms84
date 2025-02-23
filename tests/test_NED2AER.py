import numpy as np
import pandas as pd
import pytest

from transforms84.helpers import DDM2RRM
from transforms84.transforms import NED2AER

from .conftest import tol_double_atol, tol_float_atol


def test_NED2AER_raise_wrong_dtype():
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2AER(NED)  # type: ignore


def test_NED2AER_raise_wrong_dtype_unrolled():
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=np.float16)
    with pytest.raises(ValueError):
        NED2AER(NED[0], NED[1], NED[2])


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_NED2AER_point_int(dtype, tol):
    NED = np.array([[0], [1], [4]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2AER(NED),
            np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype),
            atol=tol,
        ),
    )


@pytest.mark.skip(reason="Get check data")
@pytest.mark.parametrize(
    "dtype,tol", [(np.int64, tol_double_atol), (np.int32, tol_float_atol)]
)
def test_NED2AER_points_int(dtype, tol):
    NED = np.array(
        [
            [[-9.1013], [4.1617], [4.2812]],
            [[-9.1013], [4.1617], [4.2812]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_point_unrolled(dtype, tol):
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype)
    a, e, r = NED2AER(NED[0], NED[1], NED[2])
    assert np.isclose(a, np.deg2rad(155.4271), atol=tol)
    assert np.isclose(e, np.deg2rad(-23.1609), atol=tol)
    assert np.isclose(r, 10.8849, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_point_unrolled_pandas(dtype, tol):
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype)
    df = pd.DataFrame(NED.T, columns=["N", "E", "D"])
    a, e, r = NED2AER(df["N"], df["E"], df["D"])
    assert np.isclose(a, np.deg2rad(155.4271), atol=tol)
    assert np.isclose(e, np.deg2rad(-23.1609), atol=tol)
    assert np.isclose(r, 10.8849, atol=tol)


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_point(dtype, tol):
    NED = np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype)
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_points_unrolled(dtype, tol):
    NED = np.array(
        [
            [[-9.1013], [4.1617], [4.2812]],
            [[-9.1013], [4.1617], [4.2812]],
        ],
        dtype=dtype,
    )
    a, e, r = NED2AER(
        np.ascontiguousarray(NED[:, 0, 0]),
        np.ascontiguousarray(NED[:, 1, 0]),
        np.ascontiguousarray(NED[:, 2, 0]),
    )
    assert np.all(np.isclose(a, np.deg2rad(155.4271), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(-23.1609), atol=tol))
    assert np.all(np.isclose(r, 10.8849, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_points_unrolled_pandas(dtype, tol):
    NED = np.array(
        [
            [[-9.1013], [4.1617], [4.2812]],
            [[-9.1013], [4.1617], [4.2812]],
        ],
        dtype=dtype,
    )
    df = pd.DataFrame(
        {
            "N": NED[:, 0, 0],
            "E": NED[:, 1, 0],
            "D": NED[:, 2, 0],
        }
    )
    a, e, r = NED2AER(df["N"], df["E"], df["D"])
    assert np.all(np.isclose(a, np.deg2rad(155.4271), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(-23.1609), atol=tol))
    assert np.all(np.isclose(r, 10.8849, atol=tol))


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_points(dtype, tol):
    NED = np.array(
        [
            [[-9.1013], [4.1617], [4.2812]],
            [[-9.1013], [4.1617], [4.2812]],
        ],
        dtype=dtype,
    )
    assert np.all(
        np.isclose(
            NED2AER(NED),
            DDM2RRM(np.array([[155.4271], [-23.1609], [10.8849]], dtype=dtype)),
            atol=tol,
        ),
    )


@pytest.mark.parametrize(
    "dtype,tol", [(np.float64, tol_double_atol), (np.float32, tol_float_atol)]
)
def test_NED2AER_parallel(dtype, tol):
    NED = np.ascontiguousarray(
        np.tile(np.array([[-9.1013], [4.1617], [4.2812]], dtype=dtype), 1000).T.reshape(
            (-1, 3, 1)
        )
    )
    a, e, r = NED2AER(
        np.ascontiguousarray(NED[:, 0, 0]),
        np.ascontiguousarray(NED[:, 1, 0]),
        np.ascontiguousarray(NED[:, 2, 0]),
    )
    assert np.all(np.isclose(a, np.deg2rad(155.4271), atol=tol))
    assert np.all(np.isclose(e, np.deg2rad(-23.1609), atol=tol))
    assert np.all(np.isclose(r, 10.8849, atol=tol))
