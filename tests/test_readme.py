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
    import pandas as pd

    df = pd.DataFrame(
        {
            "radLatTarget": rrm_target[:, 0, 0],
            "radLonTarget": rrm_target[:, 1, 0],
            "mAltTarget": rrm_target[:, 2, 0],
            "radLatOrigin": rrm_local[:, 0, 0],
            "radLonOrigin": rrm_local[:, 1, 0],
            "mAltOrigin": rrm_local[:, 2, 0],
        }
    )
    df["e"], df["n"], df["u"] = ECEF2ENU(
        df["radLatOrigin"],
        df["radLonOrigin"],
        df["mAltOrigin"],
        *geodetic2ECEF(
            df["radLatTarget"],
            df["radLonTarget"],
            df["mAltTarget"],
            WGS84.a,
            WGS84.b,
        ),
        WGS84.a,
        WGS84.b,
    )
    assert np.all(np.isclose(out1[:, 0, 0], df["e"]))
    assert np.all(np.isclose(out1[:, 1, 0], df["n"]))
    assert np.all(np.isclose(out1[:, 2, 0], df["u"]))


def test_example2():
    from transforms84.systems import WGS, WGS72

    assert WGS(6378135.0, 6356750.520016094) == WGS72
