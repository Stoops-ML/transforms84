#define NCOORDSIN3D 3
#define NCOORDSIN2D 2
#define DEGCIRCLE 360.0
#define PI 3.14159265358979323846

#define THREADING_CORES_MULTIPLIER 4
#define PIf 3.14159265358979323846f


PyArrayObject* get_numpy_array(PyObject* obj) {
    PyArrayObject *numpy_array = NULL;

    if (PyObject_HasAttrString(obj, "values")) {  // pandas Series
        PyArrayObject *values = (PyArrayObject*)PyObject_GetAttrString(obj, "values");
        if (values && PyArray_Check(values))
            numpy_array = values;
        else
            return NULL;
    }
    else if (PyArray_Check(obj))  // numpy array
        numpy_array = (PyArrayObject*)obj;
    else if (PyList_Check(obj)) {  // list
        PyObject* array = PyArray_FROM_OTF(obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (array && PyArray_Check(array))
            numpy_array = (PyArrayObject*)array;
        else
            return NULL;
    }
    else
        return NULL;

    return numpy_array;
}