# transforms84
![PyPI - Version](https://img.shields.io/pypi/v/transforms84)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/transforms84)
![PyPI - Downloads](https://img.shields.io/pypi/dm/transforms84)
![workflow-status](https://img.shields.io/github/actions/workflow/status/Stoops-ML/transforms84/workflow.yml)

Python library for geographic system transformations with additional helper functions.

This package focuses on:
1. Performance
2. Support for different matrix shapes:
   * Ideal point shape of `(3,1)`, `(nPoints,3,1)` (as well as `(3,)` and `(nPoints,3)`)
   * Separate input array for each axis in the coordinate system of size `(nPoints,)`
3. Functions that adapt to differing input matrices shapes: one-to-one, many-to-many and one-to-many points. See [below](#many-to-many--one-to-many) for an example.

## Installation
`pip install transforms84`

## Operations
### Coordinate Transformations
The following coordinate transformations have been implemented:
- geodetic &rarr; ECEF [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates)
- ECEF &rarr; geodetic [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates)
- ECEF &rarr; ENU [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU)
- ENU &rarr; ECEF [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF)
- ENU &rarr; AER [🔗](https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf)
- AER &rarr; ENU [🔗](https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf)
- ECEF &rarr; NED [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU) [🔗](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
- NED &rarr; ECEF [🔗](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF) [🔗](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
- NED &rarr; AER [🔗](https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf) [🔗](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
- AER &rarr; NED [🔗](https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf) [🔗](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
- geodetic &rarr; UTM [🔗](https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/)
- UTM &rarr; geodetic [🔗](https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/)

### Velocity Transformations
The following velocity transformations have been implemented:
- ECEF &rarr; NED
- NED &rarr; ECEF
- ENU &rarr; ECEF
- ECEF &rarr; ENU

### Distances
The following distance formulae have been implemented:
- Haversine [🔗](https://en.wikipedia.org/wiki/Haversine_formula#Formulation)

### Additional Functions
The following functions have been implemented:
- Angular difference (smallest and largest)
- [rad, rad, X] &rarr; [deg, deg, X]
- [deg, deg, X] &rarr; [rad, rad, X]

## Examples
See the Jupyter notebooks in [examples](examples) to see how to use the transform84. Run `pip install transforms84[examples]` to run the examples locally.

### Many-to-many & one-to-many
The `transforms.ECEF2ENU` transformation accepts same and differing matrix shape sizes. Below showcases the many-to-many method where three target points, `rrm_target`, in the geodetic coordinate system are transformed to the local ENU coordinate system about the point `rrm_local`, where both matrices are of shape (3, 3, 1):
```
>> import numpy as np
>> from transforms84.systems import WGS84
>> from transforms84.helpers import DDM2RRM
>> from transforms84.transforms import ECEF2ENU, geodetic2ECEF
>>
>> rrm_local = DDM2RRM(
>>     np.array(
>>         [[[30], [31], [0]], [[30], [31], [0]], [[30], [31], [0]]], dtype=np.float64
>>     )
>> )  # convert each point from [deg, deg, X] to [rad, rad, X]
>> rrm_target = DDM2RRM(
>>     np.array(
>>         [[[31], [32], [0]], [[31], [32], [0]], [[31], [32], [0]]], dtype=np.float64
>>     )
>> )
>> ECEF2ENU(
>>     rrm_local, geodetic2ECEF(rrm_target, WGS84.a, WGS84.b), WGS84.a, WGS84.b
>> )  # geodetic2ECEF -> ECEF2ENU
array(
    [
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
    ]
)
```

We can achieve the same result using the one-to-many method with a single local point of shape (3, 1):
```
>> rrm_local_one_point = DDM2RRM(np.array([[30], [31], [0]], dtype=np.float64))
>> ECEF2ENU(rrm_local_one_point, geodetic2ECEF(rrm_target, WGS84.a, WGS84.b), WGS84.a, WGS84.b)
array(
    [
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
        [[95499.41373564], [111272.00245298], [-1689.19916788]],
    ]
)
```

Again, we can achieve the same result by splitting the arrays over each coordiante system axis:
```
>> rad_lat_target = np.deg2rad(np.array([31, 31, 31], dtype=np.float64))
>> rad_lon_target = np.deg2rad(np.array([32, 32, 32], dtype=np.float64))
>> m_alt_target = np.array([0, 0, 0], dtype=np.float64)
>> rad_lat_origin = np.deg2rad(np.array([30, 30, 30], dtype=np.float64))
>> rad_lon_origin = np.deg2rad(np.array([31, 31, 31], dtype=np.float64))
>> m_alt_origin = np.array([0, 0, 0], dtype=np.float64)
>> ECEF2ENU(
      rad_lat_origin,
      rad_lon_origin,
      m_alt_origin,
      *geodetic2ECEF(rad_lat_target, rad_lon_target, m_alt_target, WGS84.a, WGS84.b),
      WGS84.a,
      WGS84.b,
  )
(
    array([95499.41373564, 95499.41373564, 95499.41373564]),
    array([111272.00245298, 111272.00245298, 111272.00245298]),
    array([-1689.19916788, -1689.19916788, -1689.19916788]),
)
```

### World Geodetic Systems Standards
`transforms84.systems` includes the `WGS84` class, which is the [WGS 84](https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84) version of the standard. Other standards can be created:
```
>> from transforms84.systems import WGS, WGS72
>> WGS72 == WGS(6378135.0, 6356750.520016094)
True
```

## Helpful Resources
...in no particular order:
- [Geographic coordinate conversion](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion)
- [Local tangent plane coordinates](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
- [Coordinate systems for navigation](https://www.mathworks.com/help/aerotbx/ug/coordinate-systems-for-navigation.html)
- [Fundamental coordinate system concepts](https://www.mathworks.com/help/aerotbx/ug/fundamental-coordinate-system-concepts.html)
- [Coordinate systems for modeling](https://www.mathworks.com/help/aerotbx/ug/coordinate-systems-for-modeling.html)
- [Coordinate systems for display](https://www.mathworks.com/help/aerotbx/ug/coordinate-systems-for-display.html)

## Contributing
PRs are always welcome and appreciated!

After forking the repo install the dev requirements: `pip install -e .[dev]`.
