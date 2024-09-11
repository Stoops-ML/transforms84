#ifndef GENERAL_H_   /* Include guard */
#define GENERAL_H_

#include <Python.h>
#include <numpy/arrayobject.h>
#include "general.h"

#endif // GENERAL_H_

PyArrayObject*
MakeNpyDoubleArrayFrom(PyArrayObject* array)
{
    PyArrayObject *array64 = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(array), PyArray_SHAPE(array), NPY_DOUBLE);
    if (array64 == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
        return NULL;
    }
    if (PyArray_CopyInto(array64, array) < 0) {
        Py_DECREF(array64);
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
        return NULL;
    }
    if (!(PyArray_ISCONTIGUOUS(array64))) {
        PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
        return NULL;
    }
    return array64;
}
