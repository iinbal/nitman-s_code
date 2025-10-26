#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "symnmf.h"

/**
 * @brief Convert a PyObject list of lists (matrix) into a flattened matrix.
 *
 * @param py_mat Pointer to a PyObject which is a list of lists (matrix).
 * @param numRows Number of rows in the matrix.
 * @param numCols Number of columns in the matrix.
 * @return Pointer to the matrix (flattened), or NULL if memory allocation fails.
 */
static double *pyobj_to_matrix(PyObject *py_mat, Py_ssize_t numRows, Py_ssize_t numCols){
    double *mat;
    PyObject *inner;
    int i, j;
    mat = malloc((size_t)numRows*numCols*sizeof(double));
    if(!mat) return NULL;
    for(i=0; i<numRows; i++){
        inner = PyList_GetItem(py_mat, i);
        if(!PyList_Check(inner)){
            free(mat);
            return NULL;
        }
        for(j=0; j<numCols; j++){
            mat[(i*numCols) + j] = PyFloat_AsDouble(PyList_GetItem(inner, j));
        }
    }
    return mat;
}

/**
 * @brief Convert a PyObject list of lists (datapoints) into a flattened matrix.
 *
 * @param datapoints Pointer to a PyObject which is a list of lists (datapoints).
 * @return Pointer to the matrix (flattened), or NULL if memory allocation fails.
 */
static double *datapoints_to_matrix(PyObject *datapoints){
    Py_ssize_t numRows, numCols;
    double *mat;
    if ((!PyList_Check(datapoints)) || PyList_Size(datapoints) <= 0) return NULL;
    numRows = PyList_Size(datapoints);
    numCols = PyList_Size(PyList_GetItem(datapoints, 0));
    mat = pyobj_to_matrix(datapoints, numRows, numCols);
    return mat;
}

/**
 * @brief Convert an array into a PyObject list.
 *
 * @param arr Pointer to an array.
 * @param len length of the array.
 * @return Pointer to a PyObject list or NULL if memory allocation fails.
 */
static PyObject *arr_to_list(const double *arr, int len){
    PyObject *lst, *num;
    int i;
    lst = PyList_New(len);
    if(!lst) return NULL;
    for(i=0; i<len; i++){
        num = PyFloat_FromDouble(arr[i]);
        if(!num){
            Py_XDECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, num);
    }
    return lst;
}

/**
 * @brief Convert a flattened matrix into a PyObject list of lists (matrix).
 *
 * @param mat Pointer to an array which is a flattened matrix.
 * @param numRows Number of rows in the matrix.
 * @param numCols Number of columns in the matrix.
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject *matrix_to_lists(const double *mat, int numRows, int numCols){
    PyObject *outer, *inner;
    int i;
    outer = PyList_New(numRows);
    if(!outer) return NULL;
    for(i=0; i<numRows; i++){
        inner = arr_to_list(mat + (i*numCols), numCols);
        if(!inner){
            Py_XDECREF(outer);
            return NULL;
        }
        PyList_SET_ITEM(outer, i, inner);
    }
    return outer;
}

/**
 * @brief Compute and return a matrix based on the specified option.
 *
 * @param py_datapoints Pointer to a PyObject which is a list of lists (datapoints).
 * @param option An integer flag indicating which matrix to compute: 0 ("sym"), 1 ("ddg"), 2 ("norm").
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject *shared_work(PyObject *py_datapoints, int option){
    PyObject *py_matrix=NULL;
    double *datapoints, *a_matrix=NULL, *d_matrix=NULL, *w_matrix=NULL;
    int n, d;
    datapoints = datapoints_to_matrix(py_datapoints);
    if(!datapoints) return NULL;
    n = (int)PyList_Size(py_datapoints);
    d = (int)PyList_Size(PyList_GetItem(py_datapoints, 0));
    a_matrix = build_a(datapoints, n, d);
    free(datapoints);
    if(!a_matrix) return NULL;
    if(option == 0){ /*sym*/
        py_matrix = matrix_to_lists(a_matrix, n, n);
        goto end;
    }
    d_matrix = build_d(a_matrix, n);
    if(!d_matrix) goto end; 
    if(option == 1){ /*ddg*/
        py_matrix = arr_to_list(d_matrix, n);
        goto end;
    }
    w_matrix = build_w(d_matrix, a_matrix, n); /*norm*/
    if(!w_matrix) goto end;
    py_matrix = matrix_to_lists(w_matrix, n, n);

/* Section responsible for freeing all memory and returning the desired matrix (or NULL if there was a failure) */
end: if(w_matrix) free(w_matrix); 
    if(a_matrix) free(a_matrix);
    if(d_matrix) free(d_matrix);
    return py_matrix;
}

/**
 * @brief Compute and return the H non-negative factor matrix in SymNMF.
 *
 * @param self Unused (standard Python C-API convention).
 * @param args A Python tuple containing: (py_h, py_w, n, k)
 *             - py_h: PyObject list of lists (n x k), initial H matrix.
 *             - py_w: PyObject list of lists (n x n), similarity matrix W.
 *             - n: Number of data points (rows).
 *             - k: Number of clusters (columns in H).
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject* symnmf(PyObject *self, PyObject *args){
    PyObject *py_h, *py_w, *py_matrix=NULL;
    int n,k;
    double *h_matrix=NULL, *w_matrix=NULL, *final_h=NULL;
    if(!PyArg_ParseTuple(args, "OOii", &py_h, &py_w, &n, &k)) return NULL;
    if((!PyList_Check(py_h)) || (!PyList_Check(py_w))){
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
        return NULL;
    }
    h_matrix = pyobj_to_matrix(py_h, n, k);
    w_matrix = pyobj_to_matrix(py_w, n, n);
    if((!h_matrix) || (!w_matrix)){
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
        goto end;
    }
    final_h = converge_h(h_matrix, w_matrix, n, k);
    if(!final_h){
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
        goto end;
    }
    py_matrix = matrix_to_lists(final_h, n, k);
    if(!py_matrix) PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");

/*h_matrix will be freed in converge_h*/
end: free(w_matrix);
    free(final_h);
    return py_matrix;
}

/**
 * @brief Compute and return the similarity matrix A used in SymNMF.
 *
 * @param self Unused (standard Python C-API convention).
 * @param args A PyObject list of lists (n x d), datapoints.
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject* sym(PyObject *self, PyObject *args){
    PyObject *py_datapoints, *py_matrix;
    if(!PyArg_ParseTuple(args, "O", &py_datapoints)) return NULL;
    py_matrix = shared_work(py_datapoints, 0);
    if(!py_matrix) PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
    return py_matrix;
}

/**
 * @brief Compute and return the diagonal degree matrix D used in SymNMF.
 *
 * @param self Unused (standard Python C-API convention).
 * @param args A PyObject list of lists (n x d), datapoints.
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject* ddg(PyObject *self, PyObject *args){
    PyObject *py_datapoints, *py_matrix;
    if(!PyArg_ParseTuple(args, "O", &py_datapoints)) return NULL;
    py_matrix = shared_work(py_datapoints, 1);
    if(!py_matrix) PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
    return py_matrix;
}

/**
 * @brief Compute and return the normalized similarity matrix W used in SymNMF.
 *
 * @param self Unused (standard Python C-API convention).
 * @param args A PyObject list of lists (n x d), datapoints.
 * @return Pointer to a PyObject list of lists (matrix) or NULL if memory allocation fails.
 */
static PyObject* norm(PyObject *self, PyObject *args){
    PyObject *py_datapoints, *py_matrix;
    if(!PyArg_ParseTuple(args, "O", &py_datapoints)) return NULL;
    py_matrix = shared_work(py_datapoints, 2);
    if(!py_matrix) PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred.");
    return py_matrix;
}

/*List of functions exposed by the symnmf_c Python extension module.*/
static PyMethodDef symnmfMethods[] = {
   {"sym",
        (PyCFunction)sym, METH_VARARGS,
        PyDoc_STR("Compute and return the weighted adjacency matrix A for the given datapoints.\n\n"
                  "Parameters:\n"
                  "    datapoints (list of lists of floats): An n x d matrix of datapoints.\n\n"
                  "Returns:\n"
                  "    list of lists of floats: The weighted adjacency matrix A (n x n).")
    },
    {"ddg",
        (PyCFunction)ddg, METH_VARARGS,
        PyDoc_STR("Compute and return the diagonal degree matrix D for the given datapoints.\n\n"
                  "Parameters:\n"
                  "    datapoints (list of lists of floats): An n x d matrix of datapoints.\n\n"
                  "Returns:\n"
                  "    list of floats: The diagonal elements of matrix D (length n).")
    },
    {"norm",
        (PyCFunction)norm, METH_VARARGS,
        PyDoc_STR("Compute and return the normalized similarity matrix W.\n\n"
                  "Parameters:\n"
                  "    datapoints (list of lists of floats): An n x d matrix of datapoints.\n\n"
                  "Returns:\n"
                  "    list of lists of floats: The normalized matrix W = D^(-1/2) * A * D^(-1/2) (n x n).")
    },
    {"symnmf",
        (PyCFunction)symnmf, METH_VARARGS,
        PyDoc_STR("Converge H using the SymNMF algorithm.\n\n"
                  "Parameters:\n"
                  "    h (list of lists of floats): Initial H matrix (n x k).\n"
                  "    w (list of lists of floats): Similarity matrix W (n x n).\n"
                  "    n (int): Number of datapoints.\n"
                  "    k (int): Number of clusters.\n\n"
                  "Returns:\n"
                  "    list of lists of floats: Updated H matrix after convergence (n x k).")
    },
    {NULL, NULL, 0, NULL}
};

/*Definition of the symnmf_c Python extension module.*/
static struct PyModuleDef symnmfModule = {
    PyModuleDef_HEAD_INIT,
    "symnmf_c",                                
    NULL, 
    -1,                                      
    symnmfMethods                            
};

/*Initialize the symnmf_c Python extension module.*/
PyMODINIT_FUNC PyInit_symnmf_c(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfModule);
    if (!m) return NULL;
    return m;
}