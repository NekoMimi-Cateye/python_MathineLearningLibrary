import cython
import numpy as np
cimport numpy as np

cdef extern from "sigmoid.h":
    void sigmoidForward(float *x, float *y, int data_len);
    void sigmoidBackward(float *deltaLoss, float *Loss, float *y, int data_len);

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef class Sigmoid:
    pass