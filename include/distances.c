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
            "OOOOOOd",
            &radLatStart,
            &radLonStart,
            &mAltStart,
            &radLatEnd,
            &radLonEnd,
            &mAltEnd,
            &mRadiusSphere))
        return NULL;
    if (((radLatStart = get_numpy_array(radLatStart)) == NULL) || ((radLonStart = get_numpy_array(radLonStart)) == NULL) || ((mAltStart = get_numpy_array(mAltStart)) == NULL) || ((radLatEnd = get_numpy_array(radLatEnd)) == NULL) || ((radLonEnd = get_numpy_array(radLonEnd)) == NULL) || ((mAltEnd = get_numpy_array(mAltEnd)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {radLatStart, radLonStart, mAltStart, radLatEnd, radLonEnd, mAltEnd};
    if (check_arrays_same_float_dtype(6, arrays) == 0) {
        radLatStart = (PyArrayObject *)PyArray_CastToType(radLatStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonStart = (PyArrayObject *)PyArray_CastToType(radLonStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltStart = (PyArrayObject *)PyArray_CastToType(mAltStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLatEnd = (PyArrayObject *)PyArray_CastToType(radLatEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonEnd = (PyArrayObject *)PyArray_CastToType(radLonEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltEnd = (PyArrayObject *)PyArray_CastToType(mAltEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(6, arrays) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *outRange;
    outRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLatEnd), PyArray_SHAPE(radLatEnd), PyArray_TYPE(radLatEnd));
    if (outRange == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(radLatEnd);

    // run function
    switch (PyArray_TYPE(outRange)) {
    case NPY_DOUBLE:
        HaversineDoubleUnrolled(
            (double*)PyArray_DATA(radLatStart), (double*)PyArray_DATA(radLonStart), (double*)PyArray_DATA(mAltStart), (double*)PyArray_DATA(radLatEnd), (double*)PyArray_DATA(radLonEnd), (double*)PyArray_DATA(radLatEnd), (long)nPoints, 1, mRadiusSphere, (double*)PyArray_DATA(outRange));
        break;
    case NPY_FLOAT:
        HaversineFloatUnrolled(
            (float*)PyArray_DATA(radLatStart), (float*)PyArray_DATA(radLonStart), (float*)PyArray_DATA(mAltStart), (float*)PyArray_DATA(radLatEnd), (float*)PyArray_DATA(radLonEnd), (float*)PyArray_DATA(radLatEnd), (long)nPoints, 1, (float)(mRadiusSphere), (float*)PyArray_DATA(outRange));
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
    if (!PyArg_ParseTuple(args, "OOd", &rrmStart, &rrmEnd, &mRadiusSphere))
        return NULL;
    rrmStart = get_numpy_array(rrmStart);
    rrmEnd = get_numpy_array(rrmEnd);
    if (PyErr_Occurred())
        return NULL;
    if (check_arrays_same_float_dtype(2, (PyArrayObject *[]){rrmStart, rrmEnd}) == 0) {
        rrmStart = (PyArrayObject *)PyArray_CastToType(rrmStart, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmEnd = (PyArrayObject *)PyArray_CastToType(rrmEnd, PyArray_DescrFromType(NPY_FLOAT64), 0);
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

    // prepare inputs
    npy_intp nPoints = PyArray_SIZE(rrmEnd) / NCOORDSIN3D;
    int isArraysSizeEqual = (PyArray_Size((PyObject*)rrmStart) == PyArray_Size((PyObject*)rrmEnd));
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        1, &nPoints, PyArray_TYPE(rrmEnd));
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        HaversineDoubleRolled(
            (double*)PyArray_DATA(rrmStart), (double*)PyArray_DATA(rrmEnd), (long)nPoints, isArraysSizeEqual, mRadiusSphere, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        HaversineFloatRolled(
            (float*)PyArray_DATA(rrmStart), (float*)PyArray_DATA(rrmEnd), (long)nPoints, isArraysSizeEqual, (float)(mRadiusSphere), (float*)PyArray_DATA(result_array));
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
