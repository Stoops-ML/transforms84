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
void WrapsDouble(const double* val,
    const double* maxVal,
    const double* minVal,
    long nPoints,
    int isMinValSizeOfVal,
    int isMaxValSizeOfVal,
    double* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iMin = iPoint * isMinValSizeOfVal;
        long iMax = iPoint * isMaxValSizeOfVal;
        boundedVal[iPoint] = fmod(val[iPoint] - minVal[iMin], maxVal[iMax] - minVal[iMin]) + minVal[iMin];
        if (boundedVal[iPoint] < minVal[iMin])
            boundedVal[iPoint] += maxVal[iMax] - minVal[iMin];
    }
}

/*
Wrap a point between two numbers
*/
void WrapsFloat(const float* val,
    const float* maxVal,
    const float* minVal,
    long nPoints,
    int isMinValSizeOfVal,
    int isMaxValSizeOfVal,
    float* boundedVal)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iMin = iPoint * isMinValSizeOfVal;
        long iMax = iPoint * isMaxValSizeOfVal;
        boundedVal[iPoint] = fmodf(val[iPoint] - minVal[iMin], maxVal[iMax] - minVal[iMin]) + minVal[iMin];
        if (boundedVal[iPoint] < minVal[iMin])
            boundedVal[iPoint] += maxVal[iMax] - minVal[iMin];
    }
}

static PyObject*
WrapWrapper(PyObject* self, PyObject* args)
{
    PyObject *arg1, *arg2, *arg3;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3))
        return NULL;

    // convert inputs to numpy arrays
    PyArrayObject *val = get_numpy_array(arg1);
    PyArrayObject *minVal = get_numpy_array(arg2);
    PyArrayObject *maxVal = get_numpy_array(arg3);
    if (val == NULL || minVal == NULL || maxVal == NULL)
        return NULL;
    PyArrayObject *arrays[] = {val, minVal, maxVal};
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        val = (PyArrayObject *)PyArray_CastToType(val, PyArray_DescrFromType(NPY_FLOAT64), 0);
        minVal = (PyArrayObject *)PyArray_CastToType(minVal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        maxVal = (PyArrayObject *)PyArray_CastToType(maxVal, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // checks
    long nPoints = (int)PyArray_SIZE(val);
    int isMinValSizeOfVal = (nPoints == PyArray_Size((PyObject*)minVal));
    int isMaxValSizeOfVal = (nPoints == PyArray_Size((PyObject*)maxVal));
    if (!(isMinValSizeOfVal || (PyArray_SIZE(minVal) == 1))) {
        PyErr_SetString(PyExc_ValueError, "Value and minimum value arrays must be the same size or the minimum value array must have a size of 1.");
        return NULL;
    }
    if (!(isMaxValSizeOfVal || (PyArray_SIZE(maxVal) == 1))) {
        PyErr_SetString(PyExc_ValueError, "Value and maximum value arrays must be the same size or the maximum value array must have a size of 1.");
        return NULL;
    }
    for (long i = 0; i < nPoints; ++i) {
        long iMin = i * isMinValSizeOfVal;
        long iMax = i * isMaxValSizeOfVal;
        if (PyArray_TYPE(maxVal) == NPY_DOUBLE && PyArray_TYPE(minVal) == NPY_DOUBLE) {
            if (((double*)PyArray_DATA(maxVal))[iMax] <= ((double*)PyArray_DATA(minVal))[iMin]) {
                PyErr_SetString(PyExc_ValueError, "All elements in maxVal must be greater than corresponding elements in minVal.");
                return NULL;
            }
        } else if (PyArray_TYPE(maxVal) == NPY_FLOAT && PyArray_TYPE(minVal) == NPY_FLOAT) {
            if (((float*)PyArray_DATA(maxVal))[iMax] <= ((float*)PyArray_DATA(minVal))[iMin]) {
                PyErr_SetString(PyExc_ValueError, "All elements in maxVal must be greater than corresponding elements in minVal.");
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_ValueError, "maxVal and minVal must be of the same type.");
            return NULL;
        }
    }

    // Create result array
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(val),
        PyArray_SHAPE(val),
        PyArray_TYPE(val));
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not create output array.");
        return NULL;
    }

    // run function
    if (PyArray_TYPE(val) == NPY_DOUBLE) {
        WrapsDouble((double*)PyArray_DATA(val), (double*)PyArray_DATA(maxVal), (double*)PyArray_DATA(minVal), nPoints, isMinValSizeOfVal, isMaxValSizeOfVal, (double*)PyArray_DATA(result_array));
    } else if (PyArray_TYPE(val) == NPY_FLOAT) {
        WrapsFloat((float*)PyArray_DATA(val), (float*)PyArray_DATA(maxVal), (float*)PyArray_DATA(minVal), nPoints, isMinValSizeOfVal, isMaxValSizeOfVal, (float*)PyArray_DATA(result_array));
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
