import numpy as np

from transforms84.systems import WGS, WGS72, WGS84


def test_WGS84():
    assert np.isclose(WGS84.mean_radius, 6371008.771415)
    assert np.isclose(WGS84.e, 0.08181919084296556)
    assert np.isclose(WGS84.e2, 0.006739496742333499)
    assert WGS(6378137.0, 6356752.314245) == WGS84


def test_WGS_ne():
    assert WGS72 != WGS84


def test_str_WGS():
    # TODO this is to hit 100% coverage of systems.py
    _ = str(WGS84)
    _ = repr(WGS84)
    _ = WGS84.mean_radius
    _ = WGS84.f
    _ = WGS84.f_second
    _ = WGS84.n
    _ = WGS84.e
    _ = WGS84.e2
