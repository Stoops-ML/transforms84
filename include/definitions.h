#define NCOORDSIN3D 3
#define NCOORDSIN2D 2
#define DEGCIRCLE 360.0
#define PI 3.14159265358979323846
#define PIf 3.14159265358979323846f
#define THREADING_CORES_MULTIPLIER 4

/**
 * Converts a Python object to a contiguous NumPy array.
 *
 * This function accepts various types of Python objects, including pandas Series,
 * NumPy arrays, lists, and numbers, and converts them to a contiguous NumPy array. If the 
 * (underlying) input is not a numpy array a numpy array is created of type float64.
 *
 * @param obj The Python object to convert.
 * @return A pointer to the resulting NumPy array, or NULL if the conversion fails.
 */
PyArrayObject* get_numpy_array(PyObject* obj) {
    PyArrayObject *numpy_array = NULL;

    // convert object to numpy array
    if (PyObject_HasAttrString(obj, "values")) {  // pandas Series
        PyArrayObject *values = (PyArrayObject*)PyObject_GetAttrString(obj, "values");
        if (values && PyArray_Check(values))
            numpy_array = values;
        else {
            PyErr_SetString(PyExc_TypeError, "Failed to convert pandas Series to NumPy array.");
            return NULL;
        }
    }
    else if (PyArray_Check(obj))  // numpy array
        numpy_array = (PyArrayObject*)obj;
    else if (PyList_Check(obj) || PyNumber_Check(obj)) {  // list or number
        PyObject* array = PyArray_FROM_OTF(obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (array && PyArray_Check(array))
            numpy_array = (PyArrayObject*)array;
        else {
            PyErr_SetString(PyExc_TypeError, "Failed to convert list or number to NumPy array.");
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported object type for conversion to NumPy array.");
        return NULL;
}

    // ensure numpy array is contiguous
    if (!PyArray_ISCONTIGUOUS(numpy_array)) {
        PyObject* contiguous_array = PyArray_ContiguousFromAny((PyObject*)numpy_array, NPY_FLOAT64, 0, 0);
        if (contiguous_array && PyArray_Check(contiguous_array))
            numpy_array = (PyArrayObject*)contiguous_array;
        else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to make NumPy array contiguous.");
            return NULL;
        }
    }

    return numpy_array;
}

/* Function to check if all input arrays have the same data type */
int check_arrays_same_float_dtype(int num_arrays, PyArrayObject *arrays[]) {
    PyArray_Descr *dtype = PyArray_DESCR(arrays[0]);
    for (int i = 0; i < num_arrays; ++i)
        if ((PyArray_DESCR(arrays[i]) != dtype) || !(PyArray_TYPE(arrays[i]) == NPY_FLOAT32 || PyArray_TYPE(arrays[i]) == NPY_FLOAT64))
            return 0;
    return 1;
}

/* Function to check if all input arrays have the same size */
int check_arrays_same_size(int num_arrays, PyArrayObject *arrays[]) {
    if (num_arrays < 2) {
        PyErr_SetString(PyExc_ValueError, "At least two arrays must be provided.");
        return 0;
    }
    npy_intp size = PyArray_SIZE(arrays[0]);
    for (int i = 1; i < num_arrays; ++i) {
        if (PyArray_SIZE(arrays[i]) != size) {
            PyErr_SetString(PyExc_ValueError, "Arrays must have the same size.");
            return 0;
        }
    }
    return 1;
}