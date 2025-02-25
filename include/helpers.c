#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#include "definitions.h"

/*
Calculate the angular difference between two numbers of double precision.

@param double *AngleStart array of size n angles. This is the start angle if
smallestAngle is false.
@param double *AngleEnd array of size n angles. This is the end angle if
smallestAngle is false.
@param long nAngles Number of angles in array
@param int smallestAngle Whether to calculate the angular difference between
the start and end angles or the smallest angle.
@param double Difference Angular difference
*/
double
AngularDifferenceDouble(const double AngleStart,
    const double AngleEnd,
    const double MaxValue,
    int smallestAngle)
{
    double Difference = fmod(fabs(AngleStart - AngleEnd), MaxValue);
    if (smallestAngle)
        Difference = fmin(Difference, MaxValue - Difference);
    else if (AngleStart > AngleEnd)
        Difference = MaxValue - Difference;
    return Difference;
}

/*
Calculate the angular difference between two numbers of float precision.

@param float AngleStart angle. This is the start angle if smallestAngle is
false.
@param float AngleEnd angle. This is the end angle if smallestAngle is false.
@param float MaxValue angle.
@param long nAngles Number of angles in array
@param int smallestAngle Whether to calculate the angular difference between
the start and end angles or the smallest angle.
@param float Difference Angular difference
*/
float AngularDifferenceFloat(const float AngleStart,
    const float AngleEnd,
    const float MaxValue,
    int smallestAngle)
{
    float Difference = fmodf(fabsf(AngleStart - AngleEnd), MaxValue);
    if (smallestAngle)
        Difference = fminf(Difference, MaxValue - Difference);
    else if (AngleStart > AngleEnd)
        Difference = MaxValue - Difference;
    return Difference;
}

/*
Calculate the angular difference between two numbers of float precision.

@param float *AngleStart array of size n angles. This is the start angle if
smallestAngle is false.
@param float *AngleEnd array of size n angles. This is the end angle if
smallestAngle is false.
@param long nAngles Number of angles in array
@param int smallestAngle Whether to calculate the angular difference between
the start and end angles or the smallest angle.
@param float Difference Angular difference */
void AngularDifferencesFloat(const float* AngleStart,
    const float* AngleEnd,
    const float MaxValue,
    long nAngles,
    int smallestAngle,
    float* Difference)
{
    long i;
#pragma omp parallel for if (nAngles > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (i = 0; i < nAngles; ++i) {
        Difference[i] = fmodf(fabsf(AngleStart[i] - AngleEnd[i]), MaxValue);
        if (smallestAngle)
            Difference[i] = fminf(Difference[i], MaxValue - Difference[i]);
        else if (AngleStart[i] > AngleEnd[i])
            Difference[i] = MaxValue - Difference[i];
    }
}

/*
Calculate the angular difference between two numbers of float precision.

@param double *AngleStart array of size n angles. This is the start angle if
smallestAngle is false.
@param double *AngleEnd array of size n angles. This is the end angle if
smallestAngle is false.
@param long nAngles Number of angles in array
@param int smallestAngle Whether to calculate the angular difference between
the start and end angles or the smallest angle.
@param double Difference Angular difference */
void AngularDifferencesDouble(const double* AngleStart,
    const double* AngleEnd,
    const double MaxValue,
    long nAngles,
    int smallestAngle,
    double* Difference)
{
    long i;
#pragma omp parallel for if (nAngles > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (i = 0; i < nAngles; ++i) {
        Difference[i] = fmod(fabs(AngleStart[i] - AngleEnd[i]), MaxValue);
        if (smallestAngle)
            Difference[i] = fmin(Difference[i], MaxValue - Difference[i]);
        else if (AngleStart[i] > AngleEnd[i])
            Difference[i] = MaxValue - Difference[i];
    }
}

/*
Convert a point from X, X, m to Y, Y, m in double precision

@param double *ddmPoint array of size nx3
@param long nPoints Number of angles in array
@param double *rrmPoint array of size nx3
*/
void XXM2YYMDouble(const double* rrmPoint,
    long nPoints,
    const double transform,
    double* ddmPoint)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        ddmPoint[i + 0] = rrmPoint[i + 0] * transform;
        ddmPoint[i + 1] = rrmPoint[i + 1] * transform;
        ddmPoint[i + 2] = rrmPoint[i + 2];
    }
}

/*
Convert a point from X, X, m to Y, Y, m in float precision

@param float *ddmPoint array of size nx3
@param long nPoints Number of angles in array
@param float *rrmPoint array of size nx3
*/
void XXM2YYMFloat(const float* rrmPoint,
    long nPoints,
    const float transform,
    float* ddmPoint)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        ddmPoint[i + 0] = rrmPoint[i + 0] * transform;
        ddmPoint[i + 1] = rrmPoint[i + 1] * transform;
        ddmPoint[i + 2] = rrmPoint[i + 2];
    }
}

/*
Wrap a point between two numbers
*/
void WrapsFloat3(const float* val,
    const float* maxVal,
    const float* minVal,
    long nPoints,
    float* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        boundedVal[iPoint] = fmodf(val[iPoint] - minVal[iPoint], maxVal[iPoint] - minVal[iPoint]) + minVal[iPoint];
        if (boundedVal[iPoint] < minVal[iPoint])
            boundedVal[iPoint] += maxVal[iPoint] - minVal[iPoint];
    }
}

/*
Wrap a point between two numbers
*/
void WrapsDouble3(const double* val,
    const double* maxVal,
    const double* minVal,
    long nPoints,
    double* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        boundedVal[iPoint] = fmod(val[iPoint] - minVal[iPoint], maxVal[iPoint] - minVal[iPoint]) + minVal[iPoint];
        if (boundedVal[iPoint] < minVal[iPoint])
            boundedVal[iPoint] += maxVal[iPoint] - minVal[iPoint];
    }
}

/*
Wrap a point between two numbers
*/
void WrapsFloat1(const float* val,
    const float maxVal,
    const float minVal,
    long nPoints,
    float* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        boundedVal[iPoint] = fmodf(val[iPoint] - minVal, maxVal - minVal) + minVal;
        if (boundedVal[iPoint] < minVal)
            boundedVal[iPoint] += maxVal - minVal;
    }
}

/*
Wrap a point between two numbers
*/
void WrapsDouble1(const double* val,
    const double maxVal,
    const double minVal,
    long nPoints,
    double* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        boundedVal[iPoint] = fmod(val[iPoint] - minVal, maxVal - minVal) + minVal;
        if (boundedVal[iPoint] < minVal)
            boundedVal[iPoint] += maxVal - minVal;
    }
}

/*
Wrap a point between two numbers
*/
double Wrap(const double val,
    const double maxVal,
    const double minVal)
{
    double boundedVal = fmod(val - minVal, maxVal - minVal) + minVal;
    if (boundedVal < minVal)
        boundedVal += maxVal - minVal;
    return boundedVal;
}

static PyObject*
WrapWrapper(PyObject* self, PyObject* args)
{
    PyObject *arg1, *arg2, *arg3;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3))
        return NULL;

    if (!PyArray_Check(arg1)) {
        double val, minVal, maxVal;
        if (PyLong_Check(arg1))
            val = PyLong_AsDouble(arg1);
        else if (PyFloat_Check(arg1))
            val = PyFloat_AsDouble(arg1);
        else {
            PyErr_SetString(PyExc_ValueError, "Value must be a float or an integer.");
            return NULL;
        }
        if (PyLong_Check(arg2))
            minVal = PyLong_AsDouble(arg2);
        else if (PyFloat_Check(arg2))
            minVal = PyFloat_AsDouble(arg2);
        else {
            PyErr_SetString(PyExc_ValueError, "Minimum value must be a float or an integer.");
            return NULL;
        }
        if (PyLong_Check(arg3))
            maxVal = PyLong_AsDouble(arg3);
        else if (PyFloat_Check(arg3))
            maxVal = PyFloat_AsDouble(arg3);
        else {
            PyErr_SetString(PyExc_ValueError, "Maximum value must be a float or an integer.");
            return NULL;
        }
        if (maxVal <= minVal) {
            PyErr_SetString(PyExc_ValueError, "The maximum value must be a greater than the minimum value.");
            return NULL;
        }
        return Py_BuildValue("d", Wrap(val, maxVal, +minVal));
    } else if (PyArray_Check(arg1) && (!PyArray_Check(arg2) && !PyArray_Check(arg3))) {
        PyArrayObject* val = (PyArrayObject*)arg1;

        if (!PyArray_ISCONTIGUOUS(val)) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
            return NULL;
        }
        if (PyArray_NDIM(val) != 1) {
            PyErr_SetString(PyExc_ValueError, "Input arrays have one dimensoin.");
            return NULL;
        }

        // Create result array
        PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(val),
            PyArray_SHAPE(val),
            PyArray_TYPE(val));
        if (result_array == NULL) {
            PyErr_SetString(PyExc_ValueError, "Could not create output array.");
            return NULL;
        }

        long nPoints = (int)PyArray_SIZE(val);
        double minVal, maxVal;
        if (PyLong_Check(arg2))
            minVal = PyLong_AsDouble(arg2);
        else if (PyFloat_Check(arg2))
            minVal = PyFloat_AsDouble(arg2);
        else {
            PyErr_SetString(PyExc_ValueError, "Minimum value must be a float or an integer.");
            return NULL;
        }
        if (PyLong_Check(arg3))
            maxVal = PyLong_AsDouble(arg3);
        else if (PyFloat_Check(arg3))
            maxVal = PyFloat_AsDouble(arg3);
        else {
            PyErr_SetString(PyExc_ValueError, "Maximum value must be a float or an integer.");
            return NULL;
        }
        if (maxVal <= minVal) {
            PyErr_SetString(PyExc_ValueError, "The maximum value must be a greater than the minimum value.");
            return NULL;
        }

        if (PyArray_TYPE(val) == NPY_DOUBLE)
            WrapsDouble1((double*)PyArray_DATA(val), maxVal, minVal, nPoints, (double*)PyArray_DATA(result_array));
        else if (PyArray_TYPE(val) == NPY_FLOAT)
            WrapsFloat1((float*)PyArray_DATA(val), (float)(maxVal), (float)(minVal), nPoints, (float*)PyArray_DATA(result_array));
        else {
            PyErr_SetString(PyExc_ValueError, "Only 32 and 64 bit float types accepted.");
            return NULL;
        }
        return (PyObject*)result_array;
    } else if (PyArray_Check(arg1) && PyArray_Check(arg2) && PyArray_Check(arg3)) {
        PyArrayObject* val = (PyArrayObject*)arg1;
        PyArrayObject* minVal = (PyArrayObject*)arg2;
        PyArrayObject* maxVal = (PyArrayObject*)arg3;

        if (!(PyArray_ISCONTIGUOUS(minVal)) || !(PyArray_ISCONTIGUOUS(maxVal)) || !(PyArray_ISCONTIGUOUS(val))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
            return NULL;
        }
        if ((PyArray_NDIM(minVal) != 1) || (PyArray_NDIM(maxVal) != 1) || (PyArray_NDIM(val) != 1)) {
            PyErr_SetString(PyExc_ValueError, "Input arrays have one dimensoin.");
            return NULL;
        }
        if ((PyArray_SIZE(minVal) != PyArray_SIZE(maxVal)) || (PyArray_SIZE(maxVal) != PyArray_SIZE(val))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays are of unequal size.");
            return NULL;
        }
        if ((PyArray_TYPE(minVal) != PyArray_TYPE(maxVal)) || (PyArray_TYPE(maxVal) != PyArray_TYPE(val))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must have the same type.");
            return NULL;
        }

        // Create result array
        PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(val),
            PyArray_SHAPE(val),
            PyArray_TYPE(val));
        if (result_array == NULL) {
            PyErr_SetString(PyExc_ValueError, "Could not create output array.");
            return NULL;
        }

        long nPoints = (int)PyArray_SIZE(val);
        if (PyArray_TYPE(val) == NPY_DOUBLE) {
            double* maxVals = (double*)PyArray_DATA(maxVal);
            double* minVals = (double*)PyArray_DATA(minVal);
            for (long i = 0; i < nPoints; i++) {
                if (minVals[i] > maxVals[i]) {
                    PyErr_SetString(PyExc_ValueError, "All maximum values must be larger than all minimum values.");
                    return NULL;
                }
            }
            WrapsDouble3((double*)PyArray_DATA(val), maxVals, minVals, nPoints, (double*)PyArray_DATA(result_array));
        } else if (PyArray_TYPE(val) == NPY_FLOAT) {
            float* maxVals = (float*)PyArray_DATA(maxVal);
            float* minVals = (float*)PyArray_DATA(minVal);
            for (long i = 0; i < nPoints; i++) {
                if (minVals[i] > maxVals[i]) {
                    PyErr_SetString(PyExc_ValueError, "All maximum values must be larger than all minimum values.");
                    return NULL;
                }
            }
            WrapsFloat3((float*)PyArray_DATA(val), maxVals, minVals, nPoints, (float*)PyArray_DATA(result_array));
        } else {
            PyErr_SetString(PyExc_ValueError, "Only 32 and 64 bit float types accepted.");
            return NULL;
        }
        return (PyObject*)result_array;
    } else {
        PyErr_SetString(PyExc_TypeError, "Bounds must be either two arrays or two floats and value must be a float or an array.");
        return NULL;
    }
}

static PyObject*
RadAngularDifferenceWrapper(PyObject* self, PyObject* args)
{
    PyObject *arg1, *arg2;
    int smallestAngle;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOi", &arg1, &arg2, &smallestAngle))
        return NULL;

    // Check if smallestAngle is valid
    if (!((smallestAngle == 0) || (smallestAngle == 1))) {
        PyErr_SetString(PyExc_ValueError, "Smallest angle must be True or False");
        return NULL;
    }

    // convert to numpy array
    PyArrayObject* radAngleStart = get_numpy_array(arg1);
    PyArrayObject* radAngleEnd = get_numpy_array(arg2);
    PyArrayObject *arrays[] = {radAngleStart, radAngleEnd};
    if (radAngleStart == NULL || radAngleEnd == NULL)
        return NULL;
    if (check_arrays_same_size(2, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        radAngleStart = (PyArrayObject *)PyArray_CastToType(radAngleStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radAngleEnd = (PyArrayObject *)PyArray_CastToType(radAngleEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // Create result array
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(radAngleEnd),
        PyArray_SHAPE(radAngleEnd),
        PyArray_TYPE(radAngleEnd));
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not create output array.");
        return NULL;
    }

    // run function
    long nPoints = (int)PyArray_SIZE(radAngleStart);
    if (PyArray_TYPE(radAngleEnd) == NPY_DOUBLE) {
        AngularDifferencesDouble((double*)PyArray_DATA(radAngleStart), (double*)PyArray_DATA(radAngleEnd), 2.0 * PI, nPoints, smallestAngle, (double*)PyArray_DATA(result_array));
    } else if (PyArray_TYPE(radAngleEnd) == NPY_FLOAT) {
        AngularDifferencesFloat((float*)PyArray_DATA(radAngleStart), (float*)PyArray_DATA(radAngleEnd), (float)(2.0 * PI), nPoints, smallestAngle, (float*)PyArray_DATA(result_array));
    } else {
        PyErr_SetString(PyExc_ValueError, "Only 32 and 64 bit float types accepted.");
        return NULL;
    }

    // output
    if ((nPoints == 1) && (PyArray_TYPE(result_array) == NPY_DOUBLE))
        return Py_BuildValue("d", *(double*)PyArray_DATA(result_array));
    else if ((nPoints == 1) && (PyArray_TYPE(result_array) == NPY_FLOAT))
        return Py_BuildValue("f", *(float*)PyArray_DATA(result_array));
    else
        return (PyObject*)result_array;
}

static PyObject*
DegAngularDifferenceWrapper(PyObject* self, PyObject* args)
{
    PyObject *arg1, *arg2;
    int smallestAngle;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOi", &arg1, &arg2, &smallestAngle))
        return NULL;

    // Check if smallestAngle is valid
    if (!((smallestAngle == 0) || (smallestAngle == 1))) {
        PyErr_SetString(PyExc_ValueError, "Smallest angle must be True or False");
        return NULL;
    }

    // convert to numpy array
    PyArrayObject* degAngleStart = get_numpy_array(arg1);
    PyArrayObject* degAngleEnd = get_numpy_array(arg2);
    if (degAngleStart == NULL || degAngleEnd == NULL)
        return NULL;
    PyArrayObject *arrays[] = {degAngleStart, degAngleEnd};
    if (check_arrays_same_size(2, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        degAngleStart = (PyArrayObject *)PyArray_CastToType(degAngleStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        degAngleEnd = (PyArrayObject *)PyArray_CastToType(degAngleEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // Create result array
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(degAngleEnd),
        PyArray_SHAPE(degAngleEnd),
        PyArray_TYPE(degAngleEnd));
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not create output array.");
        return NULL;
    }

    // run function
    long nPoints = (int)PyArray_SIZE(degAngleStart);
    if (PyArray_TYPE(degAngleEnd) == NPY_DOUBLE) {
        AngularDifferencesDouble((double*)PyArray_DATA(degAngleStart), (double*)PyArray_DATA(degAngleEnd), DEGCIRCLE, nPoints, smallestAngle, (double*)PyArray_DATA(result_array));
    } else if (PyArray_TYPE(degAngleEnd) == NPY_FLOAT) {
        AngularDifferencesFloat((float*)PyArray_DATA(degAngleStart), (float*)PyArray_DATA(degAngleEnd), DEGCIRCLE, nPoints, smallestAngle, (float*)PyArray_DATA(result_array));
    } else {
        PyErr_SetString(PyExc_ValueError, "Only 32 and 64 bit float types accepted.");
        return NULL;
    }

    // output
    if ((nPoints == 1) && (PyArray_TYPE(result_array) == NPY_DOUBLE))
        return Py_BuildValue("d", *(double*)PyArray_DATA(result_array));
    else if ((nPoints == 1) && (PyArray_TYPE(result_array) == NPY_FLOAT))
        return Py_BuildValue("f", *(float*)PyArray_DATA(result_array));
    else
        return (PyObject*)result_array;
}

static PyObject*
RRM2DDMWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmPoint;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rrmPoint))
        return NULL;

    // Checks
    if (!(PyArray_ISCONTIGUOUS(rrmPoint))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }

    PyArrayObject* in_array;
    if (PyArray_ISINTEGER(rrmPoint) == 0)
        in_array = rrmPoint;
    else {
        in_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmPoint), PyArray_SHAPE(rrmPoint), NPY_DOUBLE);
        if (in_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(in_array, rrmPoint) < 0) {
            Py_DECREF(in_array);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(in_array))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_SHAPE(in_array), PyArray_TYPE(in_array));
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not create output array.");
        return NULL;
    }

    long nPoints = (int)PyArray_SIZE(in_array) / NCOORDSIN3D;
    if (PyArray_TYPE(result_array) == NPY_DOUBLE) {
        XXM2YYMDouble(
            (double*)PyArray_DATA(in_array), nPoints, 180.0 / PI, (double*)PyArray_DATA(result_array));
    } else if (PyArray_TYPE(result_array) == NPY_FLOAT) {
        XXM2YYMFloat(
            (float*)PyArray_DATA(in_array), nPoints, (float)(180.0 / PI), (float*)PyArray_DATA(result_array));
    } else {
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float or int types accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
DDM2RRMWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* ddmPoint;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ddmPoint))
        return NULL;

    // Checks
    if (!(PyArray_ISCONTIGUOUS(ddmPoint))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }

    PyArrayObject* in_array;
    if (PyArray_ISINTEGER(ddmPoint) == 0)
        in_array = ddmPoint;
    else {
        in_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(ddmPoint), PyArray_SHAPE(ddmPoint), NPY_DOUBLE);
        if (in_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(in_array, ddmPoint) < 0) {
            Py_DECREF(in_array);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(in_array))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(in_array), PyArray_SHAPE(in_array), PyArray_TYPE(in_array));
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not create output array.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(in_array) / NCOORDSIN3D;
    if (PyArray_TYPE(result_array) == NPY_DOUBLE) {
        XXM2YYMDouble(
            (double*)PyArray_DATA(in_array), nPoints, PI / 180.0, (double*)PyArray_DATA(result_array));
    } else if (PyArray_TYPE(result_array) == NPY_FLOAT) {
        XXM2YYMFloat(
            (float*)PyArray_DATA(in_array), nPoints, (float)(PI / 180.0), (float*)PyArray_DATA(result_array));
    } else {
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float or int types accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//           accepting arguments, accepting keyword arguments, being a class
//           method, or being a static method of a class.
// ml_doc:  The docstring for the method
static PyMethodDef MyMethods[] = {
    { "wrap",
        WrapWrapper,
        METH_VARARGS,
        "Wrap a number in bounds" },
    { "rad_angular_difference",
        RadAngularDifferenceWrapper,
        METH_VARARGS,
        "Angular difference between two angles in radians" },
    { "deg_angular_difference",
        DegAngularDifferenceWrapper,
        METH_VARARGS,
        "Angular difference between two angles in radians" },
    { "RRM2DDM",
        RRM2DDMWrapper,
        METH_VARARGS,
        "Converts arrays of [rad, rad, m] to [deg, deg, m]" },
    { "DDM2RRM",
        DDM2RRMWrapper,
        METH_VARARGS,
        "Converts arrays of [rad, rad, m] to [deg, deg, m]" },
    { NULL, NULL, 0, NULL }
};

// Module definition
static struct PyModuleDef helpers = {
    PyModuleDef_HEAD_INIT,
    "helpers",
    "Module containing helper / miscellaneous functions",
    -1,
    MyMethods
};

// Module initialization function
PyMODINIT_FUNC
PyInit_helpers(void)
{
    import_array(); // Initialize the NumPy C API
    return PyModule_Create(&helpers);
}
