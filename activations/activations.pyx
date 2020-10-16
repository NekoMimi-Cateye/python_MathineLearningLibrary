#--------------------------------------------------#
# IMPORT LIBRARYS (PYTHON)
#--------------------------------------------------#
import cython
import numpy as np

#--------------------------------------------------#
# IMPORT LIBRARYS (CYTHON)
#--------------------------------------------------#
cimport numpy as np

#--------------------------------------------------#
# DEFINE TYPE OF NUMPY
#--------------------------------------------------#
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

#--------------------------------------------------#
# ACTIVATIONS CLASS
#--------------------------------------------------#
cdef class Sigmoid:
    cdef np.ndarray y;

    def forward(self, np.ndarray x):
        dataLen = x.size
        dataSize = x.shape()
        x = x.reshape(-1)
        self.y = np.empty(dataLen, dtype=DTYPE)
        sigmoidForward(x, y)
        return(self.y.reshape(dataSize))

    def backward(self):
        pass

#--------------------------------------------------#
# ACIVATIONS FORWARD FUNCTION
#--------------------------------------------------#
cdef sigmoidForward(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    y = 1 / (1 + np.exp(-x))