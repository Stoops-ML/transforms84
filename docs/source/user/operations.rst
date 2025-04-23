Operations
==========

Coordinate Transformations
--------------------------

The following coordinate transformations have been implemented:
- geodetic &rarr; ECEF `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates>`_
- ECEF &rarr; geodetic `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates>`_
- ECEF &rarr; ENU `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU>`_
- ENU &rarr; ECEF `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF>`_
- ENU &rarr; AER `[1] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_
- AER &rarr; ENU `[1] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_
- ECEF &rarr; NED `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU>`_ `[2] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
- NED &rarr; ECEF `[1] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF>`_ `[2] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
- NED &rarr; AER `[1] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_ `[2] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
- AER &rarr; NED `[1] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_ `[2] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
- geodetic &rarr; UTM `[1] <https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/>`_
- UTM &rarr; geodetic `[1] <https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/>`_

Velocity Transformations
------------------------

The following velocity transformations have been implemented:
- ECEF &rarr; NED
- NED &rarr; ECEF
- ENU &rarr; ECEF
- ECEF &rarr; ENU

Distances
---------

The following distance formulae have been implemented:
- Haversine `[1] <https://en.wikipedia.org/wiki/Haversine_formula#Formulation>`_

Additional Functions
--------------------

The following functions have been implemented:
- Angular difference (smallest and largest)
- [rad, rad, X] &rarr; [deg, deg, X]
- [deg, deg, X] &rarr; [rad, rad, X]