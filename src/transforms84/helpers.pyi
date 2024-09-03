import numpy.typing as npt

from . import DTYPES_SUPPORTED

def deg_angular_difference(
    angle1: float | npt.NDArray[DTYPES_SUPPORTED],
    angle2: float | npt.NDArray[DTYPES_SUPPORTED],
    smallest_angle: bool,
) -> float: ...
def rad_angular_difference(
    angle1: float | npt.NDArray[DTYPES_SUPPORTED],
    angle2: float | npt.NDArray[DTYPES_SUPPORTED],
    smallest_angle: bool,
) -> float: ...
def RRM2DDM(
    rrm_position: npt.NDArray[DTYPES_SUPPORTED],
) -> npt.NDArray[DTYPES_SUPPORTED]: ...
def DDM2RRM(
    ddm_position: npt.NDArray[DTYPES_SUPPORTED],
) -> npt.NDArray[DTYPES_SUPPORTED]: ...
