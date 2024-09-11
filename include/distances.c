#include <Python.h>
#include <numpy/arrayobject.h>
#include "definitions.h"
#include "general.h"

/*
Calculate the Haversine distance between two points in double precision.
https://en.wikipedia.org/wiki/Haversine_formula#Formulation

@param float *rrmStart array of size nx3 of start point azimuth, elevation,
range [rad, rad, m]
@param float *rrmEnd array of size nx3 of start point azimuth, elevation, range
[rad, rad, m]
@param size_t nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineDouble(const double* rrmStart,
    const double* rrmEnd,
    int nPoints,
    int isArraysSizeEqual,
    double mRadiusSphere,
    double* mDistance)
{
    int iPointEnd, iPointStart;
    for (int iPoint = 0; iPoint < nPoints; ++iPoint) {
        iPointEnd = iPoint * NCOORDSINPOINT;
        iPointStart = iPointEnd * isArraysSizeEqual;
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
@param size_t nPoints Number of target points
@param double mRadiusSphere Radius of sphere in metres
@param float *mRadiusSphere array of size nx3 of distance between start and end
points
*/
void HaversineFloat(const float* rrmStart,
    const float* rrmEnd,
    int nPoints,
    int isArraysSizeEqual,
    float mRadiusSphere,
    float* mDistance)
{
    int iPointEnd, iPointStart;
    for (int iPoint = 0; iPoint < nPoints; ++iPoint) {
        iPointEnd = iPoint * NCOORDSINPOINT;
        iPointStart = iPointEnd * isArraysSizeEqual;
        mDistance[iPoint] = (float)(2.0) * mRadiusSphere * asinf(sqrtf(((float)(1.0) - cosf(rrmEnd[iPointEnd] - rrmStart[iPointStart]) + cosf(rrmStart[iPointStart]) * cosf(rrmEnd[iPointEnd]) * ((float)(1.0) - cosf(rrmEnd[iPointEnd + 1] - rrmStart[iPointStart + 1]))) / (float)(2.0)));
    }
}

static PyObject*
HaversineWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmStart) == PyArray_NDIM(rrmEnd)) && (PyArray_SIZE(rrmStart) == PyArray_SIZE(rrmEnd)) || ((PyArray_SIZE(rrmStart) == NCOORDSINPOINT) && (PyArray_SIZE(rrmStart) < PyArray_SIZE(rrmEnd))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the start point must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmStart) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(rrmEnd) % NCOORDSINPOINT) != 0) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must be a multiple of three.");
        return NULL;
    }
    
    // ensure float types
    PyArrayObject *inArrayStart, *inArrayEnd;
    if (((PyArray_TYPE(rrmStart) == NPY_FLOAT) && (PyArray_TYPE(rrmEnd) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmStart) == 0))
        inArrayStart = MakeNpyDoubleArrayFrom(rrmStart);
    else
        inArrayStart = rrmStart;
    if (((PyArray_TYPE(rrmEnd) == NPY_FLOAT) && (PyArray_TYPE(rrmStart) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmEnd) == 0))
        inArrayEnd = MakeNpyDoubleArrayFrom(rrmEnd);
    else
        inArrayEnd = rrmEnd;

    // prepare inputs
    npy_intp nPoints = PyArray_SIZE(rrmEnd) / NCOORDSINPOINT;
    int isArraysSizeEqual = (PyArray_Size((PyObject*)inArrayStart) == PyArray_Size((PyObject*)inArrayEnd));
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        1, &nPoints, PyArray_TYPE(inArrayEnd));
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
        case NPY_DOUBLE: 
            HaversineDouble(
                (double*)PyArray_DATA(inArrayStart), (double*)PyArray_DATA(inArrayEnd), (int)nPoints, isArraysSizeEqual, mRadiusSphere, (double*)PyArray_DATA(result_array));
            break;
        case NPY_FLOAT:
            HaversineFloat(
                (float*)PyArray_DATA(inArrayStart), (float*)PyArray_DATA(inArrayEnd), (int)nPoints, isArraysSizeEqual, (float)(mRadiusSphere), (float*)PyArray_DATA(result_array));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
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
