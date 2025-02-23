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
    else if (PyObject_TypeCheck(obj, &PyArray_Type))  // numpy array
        numpy_array = (PyArrayObject*)obj;
    else
        return NULL;

    return numpy_array;
}