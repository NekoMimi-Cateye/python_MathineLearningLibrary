import cython
import numpy as np
cimport numpy as np

cdef class FullConnect:
    cdef np.ndarray dataX
    cdef np.ndarray weight
    cdef np.ndarray bias
    cdef np.ndarray deltaLossW
    cdef np.ndarray deltaLossB
    cdef function optimizer

    def __init__(self, int inLen, int outLen):
        self.dataX = np.empty((1, inLen))
        self.weight = np.empty((inLen, outLen))
        self.bias = np.empty((1, outLen))
        self.deltaLossW = np.empty((inLen, outLen))
        self.deltaLossB = np.empty((1, outLen))

    def forward(self, np.ndarray data):
        self.dataX = data
        return data@self.weight + self.bias

    def backward(self, np.ndarray loss):
        deltaLoss = loss@self.weight.T
        self.deltaLossW = self.dataX.T@loss
        self.deltaLossB = loss
        return deltaLoss

    def setOptimizer(self, function opt):
        self.optimizer = opt