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
        dataSize = x.shape
        dataLen = x.size
        self.y = np.empty(dataSize, DTYPE)
        sigmoidForward(<float *> x, <float *> self.y, dataLen)
        return(self.y)

    def backward(self):
        pass

#--------------------------------------------------#
# ACIVATIONS FORWARD FUNCTION
#--------------------------------------------------#
