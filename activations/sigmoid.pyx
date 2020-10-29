import cython
import numpy as np
cimport numpy as np

cdef class Sigmoid:
    cdef np.ndarray dataY
    cdef int yDim

    def __init__(self, tuple shape):
        self.dataY = np.empty(shape)

    def forward(self, np.ndarray data):
        self.dataY = 1 / (1 + np.exp(-data))
        return self.dataY

    def backward(self, np.ndarray loss):
        cdef np.ndarray deltaLoss
        deltaLoss = self.dataY * (1 - self.dataY) * loss
        return deltaLoss