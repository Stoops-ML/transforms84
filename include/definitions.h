#define NCOORDSIN3D 3
#define NCOORDSIN2D 2
#define DEGCIRCLE 360.0
#define PI 3.14159265358979323846

#define THREADING_CORES_MULTIPLIER 4
#define PIf 3.14159265358979323846f


PyObject* numpy_array_from_pandas_series(PyObject* obj) {
    PyObject *numpy_array = NULL;

    // Check for Pandas Series and extract the underlying numpy array
    if (PyObject_HasAttrString(obj, "values")) {
        PyObject *values = PyObject_GetAttrString(obj, "values");
        if (values && PyArray_Check(values))
            numpy_array = values;
        else
            Py_XDECREF(values); // Cleanup if it is not a numpy array
    }
    return numpy_array; // NULL if not a Series containing a numpy array
}