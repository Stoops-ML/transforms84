import numpy as np

tol_float_atol = 0.3
tol_double_atol = 0.01
tol_double_rtol = 0.001
tol_float_rtol = 0.001

float_types = [np.float32, np.float64, float]
float_type_pairs = [(t1, t2) for t1 in float_types for t2 in float_types]
