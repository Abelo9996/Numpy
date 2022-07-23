#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    /* Set initial matrices */
    Matrix61c *ourmat = (Matrix61c *) args;
    matrix *res;
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    /* Take care of dimension errors, return ValueError */
    if ((self->mat->rows != ourmat->mat->rows) || (self->mat->cols != ourmat->mat->cols)){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return NULL;
    }
    /* Take care of type errors, return TypeError */
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    /* Take care of malloc errors, returned in matrix.c */
    if (allocate_matrix(&res,self->mat->rows,self->mat->cols) == -1){
        return NULL;
    }
    finres->mat = res;
    /* Add matrices and check for any error from matrix.c */
    if (add_matrix(((Matrix61c *) finres)->mat,self->mat,ourmat->mat) == -1){
        return NULL;
    }
    /* Assign shape and return */
    finres->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject *) finres;
}

/* 
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    /* Set initial matrices */
    Matrix61c *ourmat = (Matrix61c *) args;
    matrix *res;
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    /* Take care of dimension errors, return a ValueError */
    if ((self->mat->rows != ourmat->mat->rows) || (self->mat->cols != ourmat->mat->cols)){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return NULL;
    }
    /* Check for Type errors, return a TypeError */
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    /* Allocate matrix and check for any errors from matrix.c */
    if (allocate_matrix(&res,self->mat->rows,self->mat->cols) == -1){
        return NULL;
    }
    /* Assign matrix and call for sub, check for any errors. */
    finres->mat = res;
    if (sub_matrix(((Matrix61c *) finres)->mat,self->mat,ourmat->mat) == -1){
        return NULL;
    }
    /* Assign shape and return. */
    finres->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject *) finres;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    /* Set initial matrices */
    Matrix61c *ourmat = (Matrix61c *) args;
    matrix *res;
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    /* Check for Dimension Errors, return a ValueError if found */
    if (self->mat->cols != ourmat->mat->rows){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return NULL;
    }
    /* Check for incompatible types, return a TypeError if found */
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    /* Allocate matrix and check for errors from matrix.c */
    if (allocate_matrix(&res,self->mat->rows,ourmat->mat->cols) == -1){
        return NULL;
    }
    finres->mat = res;
    /* Assign matrix and multiply matrix, check for errors */
    if (mul_matrix(((Matrix61c *) finres)->mat,self->mat,ourmat->mat) == -1){
        return NULL;
    }
    /* Set shape and return matrix. */
    finres->shape = get_shape(self->mat->rows, ourmat->mat->cols);
    return (PyObject *) finres;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    /* Set initial matrices */
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    matrix *res;
    /* Allocate space for matrix, check for any errors returned */
    if (allocate_matrix(&res,self->mat->rows,self->mat->cols) == -1){
        return NULL;
    }
    /* Set initial matrices */
    finres->mat = res;
    /* Call to neg, check if any errors come from matrix.c */
    if (neg_matrix(((Matrix61c *) finres)->mat,self->mat) == -1){
        return NULL;
    }
    /* Set matrix shape and return */
    finres->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject *) finres;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    /* Set initial matrices */
    matrix *res;
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    /* Allocate space for matrix, check for errors from matrix.c */
    if (allocate_matrix(&res,self->mat->rows,self->mat->cols) == -1){
        return NULL;
    }
    /* Set initial matrices */
    finres->mat = res;
    /* Call to abs, check for any errors */
    if (abs_matrix(((Matrix61c *) finres)->mat,self->mat) == -1){
        return NULL;
    }
    /* Set matrix shape and return */
    finres->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject *) finres;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    /* Set initial matrices */
    matrix *res;
    Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
    /* Check for incompatible types and return a TypeError if found */
    if (!PyLong_Check(pow)) {
        PyErr_SetString(PyExc_TypeError, "Power must be of type integer!");
        return NULL;
    }
    /* Check for dimensions and power value, return ValueError if found */
    if ((int) PyLong_AsLong(pow) < 0 || (self->mat->rows != self->mat->cols)) {
        PyErr_SetString(PyExc_ValueError, "Power must be non-negative!");
        return NULL;
    }
    /* Allocate space for matrix, check for any errors from matrix.c */
    if (allocate_matrix(&res,self->mat->rows,self->mat->cols) == -1){
        return NULL;
    }
    /* Set initial matrices */
    finres->mat = res;
    /* Call to pow, check for any errors */
    if (pow_matrix(((Matrix61c *) finres)->mat,self->mat,(int) PyLong_AsLong(pow)) == -1){
        return NULL;
    }
    /* Set matrix shape and return */
    finres->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject *) finres;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    Matrix61c_add,  Matrix61c_sub,  Matrix61c_multiply, 0, 0, Matrix61c_pow, Matrix61c_neg, 0, Matrix61c_abs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    /* Check for incorrect argument types, return a TypeError if found */
    if (!PyLong_Check(PyTuple_GetItem(args, 0)) || !PyLong_Check(PyTuple_GetItem(args, 1)) || (!PyLong_Check(PyTuple_GetItem(args, 2)) && !PyFloat_Check(PyTuple_GetItem(args, 2))) || PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "Please re-check your arguments!");
        return NULL;
    }
    /* Check for dimension errors, return an IndexError if found. */
    if ((int) PyLong_AsLong(PyTuple_GetItem(args, 0)) > (self->mat->rows - 1) || (int) PyLong_AsLong(PyTuple_GetItem(args, 0)) < 0 || (int) PyLong_AsLong(PyTuple_GetItem(args, 1)) > (self->mat->cols - 1) || (int) PyLong_AsLong(PyTuple_GetItem(args, 1)) < 0) {
        PyErr_SetString(PyExc_IndexError, "index values must be within the correct range!");
        return NULL;
    }
    /* Call to set and return None. */
    set(self->mat,(int) PyLong_AsLong(PyTuple_GetItem(args, 0)),(int) PyLong_AsLong(PyTuple_GetItem(args, 1)),(double) PyFloat_AsDouble(PyTuple_GetItem(args, 2)));
    Py_RETURN_NONE;
} 

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    /* Check for incompatible arguments and return a TypeError if found. */
    if (!PyLong_Check(PyTuple_GetItem(args, 0)) || !PyLong_Check(PyTuple_GetItem(args, 1)) || PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Please re-check your arguments!");
        return NULL;
    }
    /* Check for dimension errors and return IndexError if found */
    if ((int) PyLong_AsLong(PyTuple_GetItem(args, 0)) > (self->mat->rows - 1) || (int) PyLong_AsLong(PyTuple_GetItem(args, 0)) < 0 || (int) PyLong_AsLong(PyTuple_GetItem(args, 1)) > (self->mat->cols - 1) || (int) PyLong_AsLong(PyTuple_GetItem(args, 1)) < 0) {
        PyErr_SetString(PyExc_IndexError, "row and col values must be within the correct range!");
        return NULL;
    }
    /* Return the gotten value */
    return PyFloat_FromDouble(get(self->mat,(int) PyLong_AsLong(PyTuple_GetItem(args, 0)),(int) PyLong_AsLong(PyTuple_GetItem(args, 1))));
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    {"set",(PyCFunction)Matrix61c_set_value, METH_VARARGS,"Set self’s entry at the ith row and jth column to val"},{"get",(PyCFunction)Matrix61c_get_value, METH_VARARGS,"Returns the entry at the ith row and jth column"},{NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    /* Handle for tuple type arguments */
    if (PyObject_TypeCheck(key,&PyTuple_Type)){
        /* Check if both arguments are input as slices */
        if (PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PySlice_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PySlice_Type)){
        /* Set initial matrices */
        Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
        matrix *mat1;
        matrix **finmat = &mat1;
        /* Set variables to find info from slice */
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GET_ITEM(key, 0);
        j = self->mat->rows;
        /* Return error if matrix is 1d as two slices can't work */
        if (self->mat->is_1d == 1){
            PyErr_SetString(PyExc_TypeError, "1D matrices only use a single slice!");
            return NULL;     
        }
        k=x=y=z=0;
        /* Call for indices */
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        /* Check for incorrect arguments, return ValueError */
        if ((y != 1 || z ==0) && val != -1){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        /* Check for incorrect info, return a TypeError */
        } else if (val == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
        }
        /* Call for indices (indices number 2) */
        Py_ssize_t j1,k1,x1,y1,z1;
        PyObject* i1 = PyTuple_GET_ITEM(key, 1);
        j1 = self->mat->cols;
        k1=x1=y1=z1=0;
        int val1 = PySlice_GetIndicesEx(i1,j1,&k1,&x1,&y1,&z1);
        /* Check for incorrect arguments, return ValueError */
        if ((y1 != 1 || z1 ==0) && val1 != -1){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        /* Check for incorrect info, return a TypeError */
        } else if (val1 == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
        }
        /* Check if the slice index is 1, and return the float for it. */
        if ((int) z == 1 && (int) z1 == 1){
            return PyFloat_FromDouble(get(self->mat, (int) k, (int) k1));
        }
        /* Allocate space for matrix and check for any errors. */
        if (allocate_matrix_ref(finmat,self->mat,(int) k, (int) k1,(int) z,(int) z1) == -1){
            return NULL;
        }
        /* Set matrices, shape, and return. */
        finres->mat = *finmat;
        finres->shape = get_shape((int) z,(int) z1);
        return (PyObject *) finres;
    }
        /* Do not allow access if 1 column, return Index Error */
        if (self->mat->cols == 1){
            PyErr_SetString(PyExc_IndexError,"Incorrect dimension access!");
            return NULL;
        /* Check if the first argument is a integer and second is a slice */
        } else if (PyObject_TypeCheck(PyTuple_GetItem(key, 0), &PyLong_Type) && PyObject_TypeCheck(PyTuple_GetItem(key, 1), &PySlice_Type)){
        /* Set initial matrices */
        Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
        matrix *mat1;
        matrix **finmat = &mat1;
        /* Set indices variables */
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GetItem(key, 1);
        j = self->mat->cols;
        k=x=y=z=0;
        /* Call to get indices info */
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        /* Check for incorrect dimension access, return an IndexError if found */
        if ((int) PyLong_AsLong(PyTuple_GetItem(key, 0)) < 0|| PyLong_AsLong(PyTuple_GetItem(key, 0)) >= self->mat->rows){
            PyErr_SetString(PyExc_IndexError,"Incorrect dimension access!");
            return NULL;
        /* Check if the slice is 1 long, then return if True */
        } else if ((int) z == 1) {
            return PyFloat_FromDouble(get(self->mat,PyLong_AsLong(PyTuple_GetItem(key, 0)),(int) k));
        /* Check for incorrect arguments, return a ValueError */
        }else if ((y != 1 || z ==0) && val != -1){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        /* If incorrect info and incompatible type, return TypeError */
        } else if (val == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
        }
        /* Allocate matrix for space and check for errors */
        if (allocate_matrix_ref(finmat,self->mat,(int) PyLong_AsLong(PyTuple_GetItem(key, 0)),(int) k,1,(int) z) == -1){
            return NULL;
        }
        /* Set matrices,shape, and return */
        finres->mat = *finmat;
        finres->shape = get_shape(1,(int) z);
        return (PyObject *) finres;
        }else if(PyObject_TypeCheck(PyTuple_GET_ITEM(key,0),&PyLong_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key,1),&PyLong_Type)){
            if ((int) PyLong_AsLong(PyTuple_GET_ITEM(key,0)) >= self->mat->rows || (int) PyLong_AsLong(PyTuple_GET_ITEM(key,1)) >= self->mat->cols){
                PyErr_SetString(PyExc_IndexError,"Please re-check your index value!");
                return NULL;
            }else{
                return PyFloat_FromDouble(get(self->mat,(int) PyLong_AsLong(PyTuple_GET_ITEM(key,0)),(int) PyLong_AsLong(PyTuple_GET_ITEM(key,1))));
            }
        /* Same processes repeated below! */
        } else if(PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PySlice_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PyLong_Type)){
        Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
        matrix *mat1;
        matrix **finmat = &mat1;
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GetItem(key, 0);
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if ((int) PyLong_AsLong(PyTuple_GetItem(key, 1)) < 0|| PyLong_AsLong(PyTuple_GetItem(key, 1)) >= self->mat->cols){
            PyErr_SetString(PyExc_IndexError,"Incorrect dimension access!");
            return NULL;
        } else if ((int) z == 1) {
            return PyFloat_FromDouble(get(self->mat,(int) k, (int) PyLong_AsLong(PyTuple_GetItem(key, 1))));
        }else if ((y != 1 || z ==0) && val != -1){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        } else if (val == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
        }
        if (allocate_matrix_ref(finmat,self->mat,(int) k,(int) PyLong_AsLong(PyTuple_GetItem(key, 1)),(int) z,1) == -1){
            return NULL;
        }
        finres->mat = *finmat;
        finres->shape = get_shape((int) z,1);
        return (PyObject *) finres;   
        }
    }
    else if (self->mat->rows == 1 && PyObject_TypeCheck(key,&PyLong_Type)){
        if ((int) PyLong_AsLong(key) < 0 || (int) PyLong_AsLong(key)>=self->mat->cols){
            PyErr_SetString(PyExc_IndexError, "Please re-check your index value!");
            return NULL;
        }else{
            return PyFloat_FromDouble(get(self->mat,0,(int) PyLong_AsLong(key)));
        }
    } else if (PyObject_TypeCheck(key,&PyLong_Type)){
        if ((int) PyLong_AsLong(key) < 0 || (int) PyLong_AsLong(key)>=self->mat->rows){
            PyErr_SetString(PyExc_IndexError, "Please re-check your index value!");
            return NULL;
        } else if (self->mat->cols == 1){
            return PyFloat_FromDouble(get(self->mat,(int) PyLong_AsLong(key),0));
        }else{
            Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
            matrix *mat1;
            matrix **finmat = &mat1;
            if (allocate_matrix_ref(finmat,self->mat,(int) PyLong_AsLong(key),0,1,self->mat->cols) == -1){
                return NULL;
            }
            finres->mat = *finmat;
            finres->shape = get_shape(1,self->mat->cols);
            return (PyObject *) finres;
        }
    }else if (self->mat->rows == 1 && PyObject_TypeCheck(key,&PySlice_Type)){
        Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
        matrix *mat1;
        matrix **finmat = &mat1;
        Py_ssize_t j,k,x,y,z;
        PyObject* i = key;
        j = self->mat->cols;
        k=x=y=z=0;
        PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if (z == 1){
            return PyFloat_FromDouble(get(self->mat,0,(int) k));
        } else if (y != 1 || z ==0){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        }
        if (allocate_matrix_ref(finmat,self->mat,0,(int) k,1,z) == -1){
            return NULL;
        }
        finres->mat = *finmat;
        finres->shape = get_shape(1,z);
        return (PyObject *) finres;
    } else if (PyObject_TypeCheck(key,&PySlice_Type)){
        Matrix61c* finres = (Matrix61c*) Matrix61c_new(&Matrix61cType,NULL,NULL);
        matrix *mat1;
        matrix **finmat = &mat1;
        Py_ssize_t j,k,x,y,z;
        PyObject* i = key;
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if (z == 1 && val != -1){
            return PyFloat_FromDouble(get(self->mat,(int) k,0));
        } else if ((y != 1 || z ==0) && val != -1){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return NULL;
        } else if (val == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
        }
        if (allocate_matrix_ref(finmat,self->mat,(int) k,0,(int) z,self->mat->cols) == -1){
            return NULL;
        }
        finres->mat = *finmat;
        finres->shape = get_shape(z,self->mat->cols);
        return (PyObject *) finres;
    }else{
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return NULL;
    }
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    if (PyObject_TypeCheck(key, &PySlice_Type)) {
    PyObject* result = Matrix61c_subscript(self,key);
    if (result == NULL){
        return -1;
    }else{
        if (self->mat->cols == 1){
        Py_ssize_t j,k,x,y,z;
        PyObject* i = key;
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if ((!PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type)) && z==1) {
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;        
        }
        if (z == 1 && val != -1){
            set(self->mat,k,0,PyFloat_AsDouble(v));
            return 0;
        } else if ((z != PyList_GET_SIZE(v)) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return -1;
        } else if ( (!PyObject_TypeCheck(v, &PyList_Type)) ||(val == -1)){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;
        }
        for (int timer = k; timer < self->mat->cols; timer++){
            set(self->mat,timer,0,PyFloat_AsDouble(PyList_GetItem(v,timer-k)));
        }
        return 0;
        }else if (self->mat->rows == 1){
        Py_ssize_t j,k,x,y,z;
        PyObject* i = key;
        j = self->mat->cols;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if ((!PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type)) && z==1) {
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;        
        }
        if (z == 1 && val != -1){
            set(self->mat,0,k,PyFloat_AsDouble(v));
            return 0;
        } else if ((z != PyList_GET_SIZE(v)) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return -1;
        } else if ( (!PyObject_TypeCheck(v, &PyList_Type)) ||(val == -1)){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;
        }
        for (int timer = k; timer < self->mat->cols; timer++){
            set(self->mat,0,timer,PyFloat_AsDouble(PyList_GetItem(v,timer-k)));
        }
        return 0;
        }else{
        Py_ssize_t j,k,x,y,z;
        PyObject* i = key;
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if (PyList_GET_SIZE(v) != self->mat->cols && z==1) {
            PyErr_SetString(PyExc_ValueError, "Dimension Error!");
            return -1;        
        }else if (z == 1 && val != -1){
            for (int timer = 0; timer < self->mat->cols; timer++){
                PyObject* lst = PyList_GetItem(v,timer);
                if (!PyObject_TypeCheck(lst, &PyFloat_Type) && !PyObject_TypeCheck(lst, &PyLong_Type)){
                    PyErr_SetString(PyExc_TypeError, "Dimension Error!");
                    return -1;
                }
                set(self->mat,k,timer,PyFloat_AsDouble(PyList_GetItem(v,timer)));
                return 0;
            }
        } else if ((z != PyList_GET_SIZE(v)) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return -1;
        } else if (val == -1){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;
        }

        for (int timer2 = 0; timer2 < z; timer2++){
            PyObject* lst1 = PyList_GetItem(v,timer2);
            if (self->mat->cols != PyList_GET_SIZE(lst1)){
                PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
                return -1;
            }
        }
        for (int timer = k; timer < x; timer++){
            PyObject* lst = PyList_GetItem(v,timer-k);
            for (int timer1 = 0; timer1 < self->mat->cols; timer1++){
                set(self->mat,timer,timer1,PyFloat_AsDouble(PyList_GetItem(lst,timer1)));
            }
        }
        return 0;
        }
    }
    }else if (PyObject_TypeCheck(key, &PyTuple_Type)){
    PyObject* result = Matrix61c_subscript(self,key);
    if (result == NULL){
        return -1;
    }else{
        if (PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PyLong_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PySlice_Type)){
        if ((int) PyLong_AsLong(PyTuple_GET_ITEM(key, 1)) >= self->mat->cols || (int) PyLong_AsLong(PyTuple_GET_ITEM(key, 1)) < 0){
            PyErr_SetString(PyExc_IndexError, "Incorrect indexing!");
            return -1;
        }
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GET_ITEM(key, 0);
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if ((z==1 && val != -1) && !PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type)){
            PyErr_SetString(PyExc_TypeError, "Dimension Error!");
            return -1;                       
        }else if (z == 1 && val != -1){
            set(self->mat,k,(int) PyLong_AsLong(PyTuple_GET_ITEM(key, 1)),PyFloat_AsDouble(v));
            return 0;        
        } else if ((z != PyList_GET_SIZE(v)) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return -1;
        } else if (val == -1 || !PyObject_TypeCheck(v, &PyList_Type)){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;
        }
        for (int timer = k; timer < x; timer++){
            PyObject* lst = PyList_GetItem(v,timer-k);
            if (!PyObject_TypeCheck(lst, &PyLong_Type) && !PyObject_TypeCheck(lst, &PyFloat_Type)){
                PyErr_SetString(PyExc_TypeError, "Incorrect info!");
                return -1;                
            }
            set(self->mat,timer,(int) PyLong_AsLong(PyTuple_GET_ITEM(key, 1)),PyFloat_AsDouble(PyList_GetItem(v,timer-k)));
        }
        return 0;
        } else if (PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PySlice_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PySlice_Type)){
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GET_ITEM(key, 0);
        j = self->mat->rows;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        Py_ssize_t j1,k1,x1,y1,z1;
        PyObject* i1 = PyTuple_GET_ITEM(key, 1);
        j1 = self->mat->cols;
        k1=x1=y1=z1=0;
        int val1 = PySlice_GetIndicesEx(i1,j1,&k1,&x1,&y1,&z1);

        if (z == 1 && z1 == 1){
            set(self->mat,k,k1,PyFloat_AsDouble(v));
            return 0;
        }
        
        if ((z1 != PyList_GET_SIZE(v) && z == 1) || (z != PyList_GET_SIZE(v) && z1 == 1) || ((y1 != 1 || z1 ==0) && val1 != -1) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments1!");
            return -1;
        } else if ((!PyObject_TypeCheck(v, &PyList_Type)) || val == -1 || val1 == -1 || (!PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type) && z == 1 && z1 == 1)){
            PyErr_SetString(PyExc_TypeError, "Incorrect info123!");
            return -1;
        }

        if (z == 1){
            for (int timer = k1; timer < x1; timer++){
                PyObject* lst = PyList_GetItem(v,timer);
                if (!PyObject_TypeCheck(lst, &PyLong_Type) && !PyObject_TypeCheck(lst, &PyFloat_Type)){
                    PyErr_SetString(PyExc_TypeError, "Please re-check your index arguments2!");
                    return -1;
                }
                set(self->mat,k,timer,PyFloat_AsDouble(PyList_GET_ITEM(v, timer)));
            }
            return 0;
        } else if (z1 == 1){
            for (int timer = k; timer < x; timer++){
                PyObject* lst = PyList_GetItem(v,timer);
                if (!PyObject_TypeCheck(lst, &PyLong_Type) && !PyObject_TypeCheck(lst, &PyFloat_Type)){
                    PyErr_SetString(PyExc_TypeError, "Please re-check your index arguments3!");
                    return -1;
                }
                set(self->mat,timer,k1,PyFloat_AsDouble(PyList_GET_ITEM(v, timer)));
            }
            return 0;
        }
        for (int timer = k; timer < x; timer++){
            PyObject* lst = PyList_GetItem(v,timer-k);
            if (z1 != PyList_GET_SIZE(lst)){
                PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments4!");
                return -1;                
            }
            for (int timer1 = k1; timer1 < x1; timer1++){
                set(self->mat,timer,timer1,PyFloat_AsDouble(PyList_GetItem(lst,timer1-k1)));
            }
        }
        return 0;
        } else if (PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PySlice_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PyLong_Type)){
        if ((int) PyLong_AsLong(PyTuple_GET_ITEM(key, 0)) >= self->mat->rows || (int) PyLong_AsLong(PyTuple_GET_ITEM(key, 0)) < 0){
            PyErr_SetString(PyExc_IndexError, "Incorrect indexing!");
            return -1;
        }
        Py_ssize_t j,k,x,y,z;
        PyObject* i = PyTuple_GET_ITEM(key, 1);
        j = self->mat->cols;
        k=x=y=z=0;
        int val = PySlice_GetIndicesEx(i,j,&k,&x,&y,&z);
        if ((z==1 && val != -1) && !PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type)){
            PyErr_SetString(PyExc_TypeError, "Dimension Error!");
            return -1;                       
        }else if (z == 1 && val != -1){
            set(self->mat,(int) PyLong_AsLong(PyTuple_GET_ITEM(key, 0)),k,PyFloat_AsDouble(v));
            return 0;        
        } else if ((z != PyList_GET_SIZE(v)) || ((y != 1 || z ==0) && val != -1)){
            PyErr_SetString(PyExc_ValueError, "Please re-check your index arguments!");
            return -1;
        } else if (val == -1 || !PyObject_TypeCheck(v, &PyList_Type)){
            PyErr_SetString(PyExc_TypeError, "Incorrect info!");
            return -1;
        }
        for (int timer = k; timer < x; timer++){
            PyObject* lst = PyList_GetItem(v,timer-k);
            if (!PyObject_TypeCheck(lst, &PyLong_Type) && !PyObject_TypeCheck(lst, &PyFloat_Type)){
                PyErr_SetString(PyExc_TypeError, "Incorrect info!");
                return -1;                
            }
            set(self->mat,(int) PyLong_AsLong(PyTuple_GET_ITEM(key, 0)),timer,PyFloat_AsDouble(PyList_GetItem(v,timer-k)));
        }
        return 0;
        } else if (PyObject_TypeCheck(PyTuple_GET_ITEM(key, 0), &PyLong_Type) && PyObject_TypeCheck(PyTuple_GET_ITEM(key, 1), &PyLong_Type)){
            if (self->mat->cols == 1 || self->mat->rows == 1 || (!PyObject_TypeCheck(v, &PyLong_Type) && !PyObject_TypeCheck(v, &PyFloat_Type))){
                PyErr_SetString(PyExc_TypeError, "Incorrect info!");
                return -1;  
            }
            if (PyFloat_AsDouble(PyTuple_GET_ITEM(key, 0)) <0 || PyFloat_AsDouble(PyTuple_GET_ITEM(key, 1)) < 0 || PyFloat_AsDouble(PyTuple_GET_ITEM(key, 0)) >= self->mat->rows || PyFloat_AsDouble(PyTuple_GET_ITEM(key, 1)) >= self->mat->cols){
                PyErr_SetString(PyExc_IndexError, "Incorrect indexing!");
                return -1;
            }
            set(self->mat,PyFloat_AsDouble(PyTuple_GET_ITEM(key, 0)),PyFloat_AsDouble(PyTuple_GET_ITEM(key, 1)),PyFloat_AsDouble(v));
            return 0;
        }
    }
    } else if (PyObject_TypeCheck(key, &PyLong_Type)) {
    PyObject* result = Matrix61c_subscript(self,key);
    if (result == NULL){
        return -1;
    }else{
        if (self->mat->cols == 1){
            double newval = PyFloat_AsDouble(v);
            int pos = (int) PyLong_AsLong(key);
            if (!PyObject_TypeCheck(v, &PyFloat_Type) && !PyObject_TypeCheck(v, &PyLong_Type)){
                PyErr_SetString(PyExc_TypeError, "Incorrect info!");
                return -1;
            }else{
                set(self->mat,pos,0,newval);
                return 0;
            }   
        }else if (self->mat->rows == 1){
            double newval = PyFloat_AsDouble(v);
            int pos = (int) PyLong_AsLong(key);
            if (!PyObject_TypeCheck(v, &PyFloat_Type) && !PyObject_TypeCheck(v, &PyLong_Type)){
                PyErr_SetString(PyExc_TypeError, "Incorrect info!");
                return -1;
            }else{
                set(self->mat,0,pos,newval);
                return 0;
            }
        }else{
            if (PyList_GET_SIZE(v) != self->mat->cols || PyObject_TypeCheck(PyList_GetItem(v,0),&PyList_Type)){
                PyErr_SetString(PyExc_ValueError, "Incorrect info!");
                return -1;   
            }else if (PyLong_AsLong(key) < 0 || PyLong_AsLong(key) >= self->mat->rows){
                PyErr_SetString(PyExc_IndexError, "Incorrect indexing!");
                return -1;
            }else if (!PyObject_TypeCheck(v, &PyList_Type)){
                PyErr_SetString(PyExc_TypeError, "Incorrect type!");
                return -1;
            }
            for (int i = 0; i < self->mat->cols; i++){
                set(self->mat,(int) PyLong_AsLong(key),i,PyFloat_AsDouble(PyList_GetItem(v,i)));
            }
            return 0;
        }
    }
    }



}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}
