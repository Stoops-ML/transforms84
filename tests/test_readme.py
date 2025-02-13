def test_example1():
    import numpy as np

    from transforms84.helpers import DDM2RRM
    from transforms84.systems import WGS84
    from transforms84.transforms import ECEF2ENU, geodetic2ECEF

    # part 1
    rrm_local = DDM2RRM(
        np.array(
            [[[30], [31], [0]], [[30], [31], [0]], [[30], [31], [0]]], dtype=np.float64
        )
    )  # convert each point from [deg, deg, X] to [rad, rad, X]
    rrm_target = DDM2RRM(
        np.array(
            [[[31], [32], [0]], [[31], [32], [0]], [[31], [32], [0]]], dtype=np.float64
        )
    )
    out1 = ECEF2ENU(
        rrm_local, geodetic2ECEF(rrm_target, WGS84.a, WGS84.b), WGS84.a, WGS84.b
    )  # geodetic2ECEF -> ECEF2ENU
    assert np.all(
        np.isclose(
            out1,
            np.array(
                [
                    [[95499.41373564], [111272.00245298], [-1689.19916788]],
                    [[95499.41373564], [111272.00245298], [-1689.19916788]],
                    [[95499.41373564], [111272.00245298], [-1689.19916788]],
                ]
            ),
        )
    )

    # part 2
    rrm_local_one_point = DDM2RRM(np.array([[30], [31], [0]], dtype=np.float64))
    out2 = ECEF2ENU(
        rrm_local_one_point,
        geodetic2ECEF(rrm_target, WGS84.a, WGS84.b),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(out1 == out2)

    # part 3
    rad_lat_target = np.deg2rad(np.array([31, 31, 31], dtype=np.float64))
    rad_lon_target = np.deg2rad(np.array([32, 32, 32], dtype=np.float64))
    m_alt_target = np.array([0, 0, 0], dtype=np.float64)
    rad_lat_origin = np.deg2rad(np.array([30, 30, 30], dtype=np.float64))
    rad_lon_origin = np.deg2rad(np.array([31, 31, 31], dtype=np.float64))
    m_alt_origin = np.array([0, 0, 0], dtype=np.float64)
    e, n, u = ECEF2ENU(
        rad_lat_origin,
        rad_lon_origin,
        m_alt_origin,
        *geodetic2ECEF(rad_lat_target, rad_lon_target, m_alt_target, WGS84.a, WGS84.b),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(out1[:, 0, 0] == e)
    assert np.all(out1[:, 1, 0] == n)
    assert np.all(out1[:, 2, 0] == u)


def test_example2():
    from transforms84.systems import WGS, WGS72

    assert WGS(6378135.0, 6356750.520016094) == WGS72
