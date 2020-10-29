import cython
import numpy as np
cimport numpy as np

cdef class MSE:
    cdef np.ndarray dataY
    cdef np.ndarray dataT

    def __init__(self, shape):
        self.dataY = np.empty(shape)
        self.dataT = np.empty(shape)

    def forward(self, np.ndarray dataY, np.ndarray dataT):
        self.dataY = dataY
        self.dataT = dataT
        return np.average(0.5 * (dataY - dataT) * (dataY - dataT))

    def backward(self):
        return self.dataY - self.dataT