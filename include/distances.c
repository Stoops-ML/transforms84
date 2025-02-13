#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#include "definitions.h"

/*
Calculate the Haversine distance between two points in double precision.
https://en.wikipedia.org/wiki/Haversine_formula#Formulation

@param float *rrmStart array of size nx3 of start point azimuth, elevation,
range [rad, rad, m]
@param float *rrmEnd array of size nx3 of start point azimuth, elevation, range
[rad, rad, m]
@param long nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineDoubleUnrolled(const double* radLatStart,
    const double* radLongStart,
    const double* mAltStart,
    const double* radLatEnd,
    const double* radLongEnd,
    const double* mAltEnd,
    long nPoints,
    int isArraysSizeEqual,
    double mRadiusSphere,
    double* mDistance)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iPointStart = iPoint * isArraysSizeEqual;
        mDistance[iPoint] = 2.0 * mRadiusSphere * asin(sqrt((1.0 - cos(radLatEnd[iPoint] - radLatStart[iPoint]) + cos(radLatStart[iPoint]) * cos(radLatEnd[iPoint]) * (1.0 - cos(radLongEnd[iPoint] - radLongStart[iPoint]))) / 2.0));
    }
}

/*
Calculate the Haversine distance between two points in float precision.
https://en.wikipedia.org/wiki/Haversine_formula#Formulation

@param float *rrmStart array of size nx3 of start point azimuth, elevation,
range [rad, rad, m]
@param float *rrmEnd array of size nx3 of start point azimuth, elevation, range
[rad, rad, m]
@param long nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineFloatUnrolled(const float* radLatStart,
    const float* radLongStart,
    const float* mAltStart,
    const float* radLatEnd,
    const float* radLongEnd,
    const float* mAltEnd,
    long nPoints,
    int isArraysSizeEqual,
    float mRadiusSphere,
    float* mDistance)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iPointEnd = iPoint * NCOORDSIN3D;
        long iPointStart = iPointEnd * isArraysSizeEqual;
        mDistance[iPoint] = (float)(2.0) * mRadiusSphere * asinf(sqrtf(((float)(1.0) - cosf(radLatEnd[iPoint] - radLatStart[iPoint]) + cosf(radLatStart[iPoint]) * cosf(radLatEnd[iPoint]) * ((float)(1.0) - cosf(radLongEnd[iPoint] - radLongStart[iPoint]))) / (float)(2.0)));
    }
}

/*
Calculate the Haversine distance between two points in double precision.
https://en.wikipedia.org/wiki/Haversine_formula#Formulation

@param float *rrmStart array of size nx3 of start point azimuth, elevation,
range [rad, rad, m]
@param float *rrmEnd array of size nx3 of start point azimuth, elevation, range
[rad, rad, m]
@param long nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineDoubleRolled(const double* rrmStart,
    const double* rrmEnd,
    long nPoints,
    int isArraysSizeEqual,
    double mRadiusSphere,
    double* mDistance)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iPointEnd = iPoint * NCOORDSIN3D;
        long iPointStart = iPointEnd * isArraysSizeEqual;
        mDistance[iPoint] = 2.0 * mRadiusSphere * asin(sqrt((1.0 - cos(rrmEnd[iPointEnd] - rrmStart[iPointStart]) + cos(rrmStart[iPointStart]) * cos(rrmEnd[iPointEnd]) * (1.0 - cos(rrmEnd[iPointEnd + 1] - rrmStart[iPointStart + 1]))) / 2.0));
    }
}

/*
Calculate the Haversine distance between two points in float precision.
https://en.wikipedia.org/wiki/Haversine_formula#Formulation

@param float *rrmStart array of size nx3 of start point azimuth, elevation,
range [rad, rad, m]
@param float *rrmEnd array of size nx3 of start point azimuth, elevation, range
[rad, rad, m]
@param long nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineFloatRolled(const float* rrmStart,
    const float* rrmEnd,
    long nPoints,
    int isArraysSizeEqual,
    float mRadiusSphere,
    float* mDistance)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long iPointEnd = iPoint * NCOORDSIN3D;
        long iPointStart = iPointEnd * isArraysSizeEqual;
        mDistance[iPoint] = (float)(2.0) * mRadiusSphere * asinf(sqrtf(((float)(1.0) - cosf(rrmEnd[iPointEnd] - rrmStart[iPointStart]) + cosf(rrmStart[iPointStart]) * cosf(rrmEnd[iPointEnd]) * ((float)(1.0) - cosf(rrmEnd[iPointEnd + 1] - rrmStart[iPointStart + 1]))) / (float)(2.0)));
    }
}

static PyObject*
HaversineUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatStart, *radLonStart, *mAltStart, *radLatEnd, *radLonEnd, *mAltEnd;
    double mRadiusSphere;

    // checks
    if (!PyArg_ParseTuple(args,
            "O!O!O!O!O!O!d",
            &PyArray_Type,
            &radLatStart,
            &PyArray_Type,
            &radLonStart,
            &PyArray_Type,
            &mAltStart,
            &PyArray_Type,
            &radLatEnd,
            &PyArray_Type,
            &radLonEnd,
            &PyArray_Type,
            &mAltEnd,
            &mRadiusSphere))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(radLatStart)) || !(PyArray_ISCONTIGUOUS(radLonStart)) || !(PyArray_ISCONTIGUOUS(mAltStart)) || !(PyArray_ISCONTIGUOUS(radLatEnd)) || !(PyArray_ISCONTIGUOUS(radLonEnd)) || !(PyArray_ISCONTIGUOUS(mAltEnd))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatStart) == PyArray_SIZE(radLonStart)) && (PyArray_SIZE(radLatStart) == PyArray_SIZE(mAltStart)) && (PyArray_SIZE(radLatStart) == PyArray_SIZE(radLatEnd)) && (PyArray_SIZE(radLatStart) == PyArray_SIZE(radLonEnd)) && (PyArray_SIZE(radLatStart) == PyArray_SIZE(mAltEnd)))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have the same size.");
        return NULL;
    }

    // ensure matching floating point types
    PyArrayObject *inArrayLatStart, *inArrayLonStart, *inArrayAltStart, *inArrayLatEnd, *inArrayLonEnd, *inArrayAltEnd;
    if (PyArray_ISINTEGER(radLatStart) == 0)
        inArrayLatStart = radLatStart;
    else {
        inArrayLatStart = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatStart), PyArray_SHAPE(radLatStart), NPY_DOUBLE);
        if (inArrayLatStart == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatStart, radLatStart) < 0) {
            Py_DECREF(inArrayLatStart);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatStart))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonStart) == 0)
        inArrayLonStart = radLonStart;
    else {
        inArrayLonStart = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonStart), PyArray_SHAPE(radLonStart), NPY_DOUBLE);
        if (inArrayLonStart == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonStart, radLonStart) < 0) {
            Py_DECREF(inArrayLonStart);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonStart))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltStart) == 0)
        inArrayAltStart = mAltStart;
    else {
        inArrayAltStart = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltStart), PyArray_SHAPE(mAltStart), NPY_DOUBLE);
        if (inArrayAltStart == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltStart, mAltStart) < 0) {
            Py_DECREF(inArrayAltStart);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltStart))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatEnd) == 0)
        inArrayLatEnd = radLatEnd;
    else {
        inArrayLatEnd = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatEnd), PyArray_SHAPE(radLatEnd), NPY_DOUBLE);
        if (inArrayLatEnd == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatEnd, radLatEnd) < 0) {
            Py_DECREF(inArrayLatEnd);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatEnd))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonEnd) == 0)
        inArrayLonEnd = radLonEnd;
    else {
        inArrayLonEnd = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonEnd), PyArray_SHAPE(radLonEnd), NPY_DOUBLE);
        if (inArrayLonEnd == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonEnd, radLonEnd) < 0) {
            Py_DECREF(inArrayLonEnd);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonEnd))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltEnd) == 0)
        inArrayAltEnd = mAltEnd;
    else {
        inArrayAltEnd = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltEnd), PyArray_SHAPE(mAltEnd), NPY_DOUBLE);
        if (inArrayAltEnd == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltEnd, mAltEnd) < 0) {
            Py_DECREF(inArrayAltEnd);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltEnd))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *outRange;
    outRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAltEnd), PyArray_SHAPE(inArrayAltEnd), PyArray_TYPE(inArrayAltEnd));
    if (outRange == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayAltEnd);

    // run function
    switch (PyArray_TYPE(outRange)) {
    case NPY_DOUBLE:
        HaversineDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatStart), (double*)PyArray_DATA(inArrayLonStart), (double*)PyArray_DATA(inArrayAltStart), (double*)PyArray_DATA(inArrayLatEnd), (double*)PyArray_DATA(inArrayLonEnd), (double*)PyArray_DATA(inArrayAltEnd), (long)nPoints, 1, mRadiusSphere, (double*)PyArray_DATA(outRange));
        break;
    case NPY_FLOAT:
        HaversineFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatStart), (float*)PyArray_DATA(inArrayLonStart), (float*)PyArray_DATA(inArrayAltStart), (float*)PyArray_DATA(inArrayLatEnd), (float*)PyArray_DATA(inArrayLonEnd), (float*)PyArray_DATA(inArrayAltEnd), (long)nPoints, 1, (float)(mRadiusSphere), (float*)PyArray_DATA(outRange));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)outRange;
}

static PyObject*
HaversineRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmStart, *rrmEnd;
    double mRadiusSphere;

    // checks
    if (!PyArg_ParseTuple(args,
            "O!O!d",
            &PyArray_Type,
            &rrmStart,
            &PyArray_Type,
            &rrmEnd,
            &mRadiusSphere))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmStart)) || !(PyArray_ISCONTIGUOUS(rrmEnd))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if (!((PyArray_NDIM(rrmStart) == PyArray_NDIM(rrmEnd)) && (PyArray_SIZE(rrmStart) == PyArray_SIZE(rrmEnd)) || ((PyArray_SIZE(rrmStart) == NCOORDSIN3D) && (PyArray_SIZE(rrmStart) <= PyArray_SIZE(rrmEnd))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the start point must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmStart) % NCOORDSIN3D) != 0 || (PyArray_SIZE(rrmEnd) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must be a multiple of three.");
        return NULL;
    }

    // ensure matching floating point types
    PyArrayObject *inArrayStart, *inArrayEnd;
    if (((PyArray_TYPE(rrmStart) == NPY_FLOAT) && (PyArray_TYPE(rrmEnd) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmStart) == 0)) {
        inArrayStart = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmStart), PyArray_SHAPE(rrmStart), NPY_DOUBLE);
        if (inArrayStart == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayStart, rrmStart) < 0) {
            Py_DECREF(inArrayStart);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayStart))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayStart = rrmStart;
    if (((PyArray_TYPE(rrmEnd) == NPY_FLOAT) && (PyArray_TYPE(rrmStart) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmEnd) == 0)) {
        inArrayEnd = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmEnd), PyArray_SHAPE(rrmEnd), NPY_DOUBLE);
        if (inArrayEnd == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayEnd, rrmEnd) < 0) {
            Py_DECREF(inArrayEnd);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayEnd))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayEnd = rrmEnd;

    // prepare inputs
    npy_intp nPoints = PyArray_SIZE(rrmEnd) / NCOORDSIN3D;
    int isArraysSizeEqual = (PyArray_Size((PyObject*)inArrayStart) == PyArray_Size((PyObject*)inArrayEnd));
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        1, &nPoints, PyArray_TYPE(inArrayEnd));
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        HaversineDoubleRolled(
            (double*)PyArray_DATA(inArrayStart), (double*)PyArray_DATA(inArrayEnd), (long)nPoints, isArraysSizeEqual, mRadiusSphere, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        HaversineFloatRolled(
            (float*)PyArray_DATA(inArrayStart), (float*)PyArray_DATA(inArrayEnd), (long)nPoints, isArraysSizeEqual, (float)(mRadiusSphere), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    if ((nPoints == 1) && (PyArray_NDIM(rrmEnd) == 2) && (PyArray_TYPE(result_array) == NPY_DOUBLE))
        return Py_BuildValue("d", *((double*)PyArray_DATA(result_array)));
    else if ((nPoints == 1) && (PyArray_NDIM(rrmEnd) == 2) && (PyArray_TYPE(result_array) == NPY_FLOAT))
        return Py_BuildValue("f", *((float*)PyArray_DATA(result_array)));
    else
        return (PyObject*)result_array;
}

static PyObject*
HaversineWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 3)
        return HaversineRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 7)
        return HaversineUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either three or seven inputs");
        return NULL;
    }
}


// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//           accepting arguments, accepting keyword arguments, being a class
//           method, or being a static method of a class.
// ml_doc:  The docstring for the method
static PyMethodDef MyMethods[] = {
    { "Haversine", HaversineWrapper, METH_VARARGS, "Haversine function" },
    { NULL, NULL, 0, NULL }
};

// Module definition
static struct PyModuleDef distances = {
    PyModuleDef_HEAD_INIT,
    "distances",
    "Module that contains functions to calculate distances between points",
    -1,
    MyMethods
};

// Module initialization function
PyMODINIT_FUNC
PyInit_distances(void)
{
    import_array(); // Initialize the NumPy C API
    return PyModule_Create(&distances);
}
