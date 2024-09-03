#include <Python.h>
#include <numpy/arrayobject.h>
#define NCOORDSINPOINT 3

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
        mDistance[iPoint] = 2.0 * mRadiusSphere * asin(sqrt((1.0 - cos(rrmEnd[iPointEnd] - rrmStart[iPointStart]) + cos(rrmStart[iPointStart]) * cos(rrmEnd[iPointEnd]) * (1.0 - cos(rrmEnd[iPointEnd + 1] - rrmStart[iPointStart + 1]))) / 2.0));
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
    if (PyArray_TYPE(rrmStart) != PyArray_TYPE(rrmEnd)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same type.");
        return NULL;
    }

    npy_intp nPoints = PyArray_SIZE(rrmEnd) / NCOORDSINPOINT;
    int array_type;
    PyArrayObject *inArrayEnd, *inArrayStart;
    if (PyArray_ISINTEGER(rrmEnd) == 0) {
        array_type = PyArray_TYPE(rrmEnd);
        inArrayStart = rrmStart;
        inArrayEnd = rrmEnd;
    } else {
        array_type = NPY_FLOAT;
        inArrayEnd = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmEnd), PyArray_SHAPE(rrmEnd), array_type);
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
        inArrayStart = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmStart), PyArray_SHAPE(rrmStart), array_type);
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
    }
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        1, &nPoints, array_type);
    int isArraysSizeEqual = (PyArray_Size((PyObject*)inArrayStart) == PyArray_Size((PyObject*)inArrayEnd));
    if (result_array == NULL)
        return NULL;
    if (PyArray_TYPE(result_array) == NPY_DOUBLE) {
        double* data1 = (double*)PyArray_DATA(inArrayStart);
        double* data2 = (double*)PyArray_DATA(inArrayEnd);
        double* result_data = (double*)PyArray_DATA((PyArrayObject*)result_array);
        HaversineDouble(
            data1, data2, (int)nPoints, isArraysSizeEqual, mRadiusSphere, result_data);
    } else if (PyArray_TYPE(result_array) == NPY_FLOAT) {
        float* data1 = (float*)PyArray_DATA(inArrayStart);
        float* data2 = (float*)PyArray_DATA(inArrayEnd);
        float* result_data = (float*)PyArray_DATA((PyArrayObject*)result_array);
        HaversineFloat(
            data1, data2, (int)nPoints, isArraysSizeEqual, mRadiusSphere, result_data);
    } else {
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types accepted.");
        return NULL;
    }
    return result_array;
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
