Operations
==========

Coordinate Transformations
--------------------------

The following coordinate transformations have been implemented:
* geodetic → ECEF `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates>`_
* ECEF → geodetic `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates>`_
* ECEF → ENU `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU>`_
* ENU → ECEF `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF>`_
* ENU → AER `[🔗] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_
* AER → ENU `[🔗] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_
* ECEF → NED `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU>`_ `[🔗] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
* NED → ECEF `[🔗] <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF>`_ `[🔗] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
* NED → AER `[🔗] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_ `[🔗] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
* AER → NED `[🔗] <https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf>`_ `[🔗] <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
* geodetic → UTM `[🔗] <https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/>`_
* UTM → geodetic `[🔗] <https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/>`_

Velocity Transformations
------------------------

The following velocity transformations have been implemented:
* ECEF → NED
* NED → ECEF
* ENU → ECEF
* ECEF → ENU

Distances
---------

The following distance formulae have been implemented:
* Haversine `[🔗] <https://en.wikipedia.org/wiki/Haversine_formula#Formulation>`_

Additional Functions
--------------------

The following functions have been implemented:
* Angular difference (smallest and largest)
* [rad, rad, X] → [deg, deg, X]
* [deg, deg, X] → [rad, rad, X]