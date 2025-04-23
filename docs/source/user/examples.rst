Examples
========

Making a square
---------------

In this example we'll be making a square. Firstly, import the following packages::

    >>> import numpy as np
    >>> import shapely
    >>> from czml3 import Document, Packet, Preamble
    >>> from czml3.properties import Color, Material, Polygon, PositionList, SolidColorMaterial
    >>> from transforms84.helpers import DDM2RRM, RRM2DDM
    >>> from transforms84.systems import WGS84
    >>> from transforms84.transforms import (
    >>>     AER2ENU,
    >>>     ENU2ECEF,
    >>>     ECEF2geodetic,
    >>> )

We'll be defining the distance between points as 5,000 metres and setting the top left point of the box at 34 deg latitude, 33 deg longitude and 0 metres altitude::

    >>> m_dist = 5000.0
    >>> rrm_top_left = DDM2RRM(np.array([[34], [33], [0]], dtype=np.float64))

The top right point::

    >>> rrm_top_right = ECEF2geodetic(
    >>>     ENU2ECEF(
    >>>         rrm_top_left,
    >>>         AER2ENU(DDM2RRM(np.array([[90], [0], [m_dist]], dtype=np.float64))),
    >>>         WGS84.a,
    >>>         WGS84.b,
    >>>     ),
    >>>     WGS84.a,
    >>>     WGS84.b,
    >>> )

The bottom left point::

    >>> rrm_bottom_left = ECEF2geodetic(
    >>>     ENU2ECEF(
    >>>         rrm_top_left,
    >>>         AER2ENU(DDM2RRM(np.array([[180], [0], [m_dist]], dtype=np.float64))),
    >>>         WGS84.a,
    >>>         WGS84.b,
    >>>     ),
    >>>     WGS84.a,
    >>>     WGS84.b,
    >>> )

Finally, the bottom right point::

    >>> rrm_bottom_right = ECEF2geodetic(
    >>>     ENU2ECEF(
    >>>         rrm_bottom_left,
    >>>         AER2ENU(DDM2RRM(np.array([[90], [0], [m_dist]], dtype=np.float64))),
    >>>         WGS84.a,
    >>>         WGS84.b,
    >>>     ),
    >>>     WGS84.a,
    >>>     WGS84.b,
    >>> )

We can print the shape using Shapely::

    >>> shapely.Polygon(
    >>>     [
    >>>         p[[1, 0]]
    >>>         for p in [rrm_top_left, rrm_top_right, rrm_bottom_right, rrm_bottom_left]
    >>>     ]
    >>> )

or using ``czml3``::

    >>> rrm_points = (
    >>>     np.hstack([rrm_top_left, rrm_top_right, rrm_bottom_right, rrm_bottom_left])
    >>>     .T[:, [1, 0, 2]]
    >>>     .ravel()
    >>>     .tolist()
    >>> )
    >>> packets = [
    >>>     Packet(id="document", name="square", version=CZML_VERSION),
    >>>     Packet(
    >>>         polygon=Polygon(
    >>>             positions=PositionList(cartographicRadians=rrm_points),
    >>>             material=Material(
    >>>                 solidColor=SolidColorMaterial(color=Color(rgba=[255, 253, 55, 255]))
    >>>             ),
    >>>             outline=True,
    >>>         ),
    >>>     ),
    >>> ]
    >>> with open("square.czml", "w") as f:
    >>>     f.write(Document(packets=packets).dumps())


Many-to-many, one-to-many and splitting inputs over axes
--------------------------------------------------------

The transformation functions accepts same and differing matrix shape sizes, as well a differing number of inputs. Below showcases the many-to-many method where three target points, rrm_target, in the geodetic coordinate system are transformed to the local ENU coordinate system about the point rrm_local, where both matrices are of shape ``(3, 3, 1)``. First import the following pacakges::

    >>> import numpy as np
    >>> import pandas as pd
    >>> from transforms84.systems import WGS84
    >>> from transforms84.helpers import DDM2RRM
    >>> from transforms84.transforms import ECEF2ENU, geodetic2ECEF

We'll create the local and target points::

    >>> rrm_local = DDM2RRM(
    >>>     np.array(
    >>>         [[[30], [31], [0]], [[30], [31], [0]], [[30], [31], [0]]], dtype=np.float64
    >>>     )
    >>> )  # convert each point from [deg, deg, X] to [rad, rad, X]
    >>> rrm_target = DDM2RRM(
    >>>     np.array(
    >>>         [[[31], [32], [0]], [[31], [32], [0]], [[31], [32], [0]]], dtype=np.float64
    >>>     )
    >>> )

Then we can convert each target point from ECEF to ENU about each local point as so (the many-to-many method)::

    >>> ECEF2ENU(
    >>>     rrm_local, geodetic2ECEF(rrm_target, WGS84.a, WGS84.b), WGS84.a, WGS84.b
    >>> )  # geodetic2ECEF -> ECEF2ENU
    array(
        [
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
        ]
    )

We can achieve the same result using the one-to-many method with a single local point of shape ``(3, 1)``::

    >>> rrm_local_one_point = DDM2RRM(np.array([[30], [31], [0]], dtype=np.float64))
    >>> ECEF2ENU(rrm_local_one_point, geodetic2ECEF(rrm_target, WGS84.a, WGS84.b), WGS84.a, WGS84.b)
    array(
        [
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
            [[95499.41373564], [111272.00245298], [-1689.19916788]],
        ]
    )


We can achieve the same result by splitting the arrays over each coordiante system axis::

    >>> df = pd.DataFrame(
    >>>    {
    >>>        "radLatTarget": rrm_target[:, 0, 0],
    >>>        "radLonTarget": rrm_target[:, 1, 0],
    >>>        "mAltTarget": rrm_target[:, 2, 0],
    >>>    }
    >>> )
    >>> df["e"], df["n"], df["u"] = ECEF2ENU(
    >>>    np.deg2rad(30),
    >>>    np.deg2rad(31),
    >>>    0,
    >>>    *geodetic2ECEF(
    >>>        df["radLatTarget"],
    >>>        df["radLonTarget"],
    >>>        df["mAltTarget"],
    >>>        WGS84.a,
    >>>        WGS84.b,
    >>>    ),
    >>>    WGS84.a,
    >>>    WGS84.b,
    >>> )
    >>> df[["e", "n", "u"]]
                  e              n            u
    0  95499.413736  111272.002453 -1689.199168
    1  95499.413736  111272.002453 -1689.199168
    2  95499.413736  111272.002453 -1689.199168


World geodetic systems standards
--------------------------------

``transforms84.systems`` includes the ``WGS84`` class, which is the `WGS 84 <https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84>`_ version of the standard. Other standards can be created::

    >>> from transforms84.systems import WGS, WGS72
    >>> WGS72 == WGS(6378135.0, 6356750.520016094)
    True

Distance metrics
----------------
Firstly, import the following packages::

    >>> import numpy as np
    >>> from transforms84.distances import Haversine
    >>> from transforms84.helpers import DDM2RRM
    >>> from transforms84.systems import WGS84

We define local and target points::

    >>> rrm_local = DDM2RRM(np.array([[30], [31], [0]], dtype=np.float64))
    >>> rrm_target = DDM2RRM(np.array([[31], [32], [0]], dtype=np.float64))

Using the Haversine formula to calculate the distance::

    >>> Haversine(rrm_local, rrm_target, WGS84.mean_radius)
    146775.88330369865

We can also use the many-to-many method::

    >>> Haversine(
    >>>     np.ascontiguousarray(np.tile(rrm_local, 10).T.reshape((-1, 3, 1))),
    >>>     np.ascontiguousarray(np.tile(rrm_target, 10).T.reshape((-1, 3, 1))),
    >>>     WGS84.mean_radius,
    >>> )
    array([146775.8833037, 146775.8833037, 146775.8833037, 146775.8833037,
        146775.8833037, 146775.8833037, 146775.8833037, 146775.8833037,
        146775.8833037, 146775.8833037])

And the one-to-many method::

    >>> Haversine(
    >>>     rrm_local,
    >>>     np.ascontiguousarray(np.tile(rrm_target, 10).T.reshape((-1, 3, 1))),
    >>>     WGS84.mean_radius,
    >>> )
    array([146775.8833037, 146775.8833037, 146775.8833037, 146775.8833037,
        146775.8833037, 146775.8833037, 146775.8833037, 146775.8833037,
        146775.8833037, 146775.8833037])
    

Angular differences
-------------------

Firstly, import the following packages::

    >>> from transforms84.helpers import deg_angular_difference

We can calculate the angular difference using the smallest angle as so::

    >>> deg_angular_difference(50, 70, True), deg_angular_difference(70, 50, True)
    (20.0, 20.0)

We can also get the actual angle (i.e. not the smallest angle)::

    >>> deg_angular_difference(50, 70, False), deg_angular_difference(70, 50, False)
    (20.0, 340.0)

The functions also accept arrays as inputs::

    >>> (
    >>>     deg_angular_difference(
    >>>         np.array([50, 50], dtype=np.float32), np.array([70, 70], dtype=np.float32), True
    >>>     ),
    >>>     deg_angular_difference(
    >>>         np.array([70, 70], dtype=np.float32), np.array([50, 50], dtype=np.float32), True
    >>>     ),
    >>>     deg_angular_difference(
    >>>         np.array([70, 70], dtype=np.float32),
    >>>         np.array([50, 50], dtype=np.float32),
    >>>         False,
    >>>     ),
    >>> )
    (array([20., 20.], dtype=float32),
    array([20., 20.], dtype=float32),
    array([340., 340.], dtype=float32))