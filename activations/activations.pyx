import cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

class Sigmoid:
    def __init__(self, shape):
        self.x = np.zeros(shape, dtype = np.float32)
        self.y = np.zeros(shape, dtype = np.float32)
        self.deltaLoss_x = np.zeros(shape, dtype = np.float32)
        self.deltaLoss_y = np.zeros(shape, dtype = np.float32)

    def forward(self, x):
        self.x = x
        self.xLen = len(x)
        sigmoidForward(x, self.y)
        return self.y

    def backward(self, deltaLoss):
        self.deltaLoss_y = deltaLoss
        sigmoidBackward(self.deltaLoss_x, self.deltaLoss_y, self.y)
        return self.deltaLoss_x

#------------------------------------------------------------#
# FUNCTION: Sigmoid
#------------------------------------------------------------#
cdef void sigmoidForward(np.ndarray[DTYPE_t] x, np.ndarray[DTYPE_t] y):
    y = 1 / (1 + np.exp(-x))

cdef void sigmoidBackward(np.ndarray[DTYPE_t] deltaLossX, np.ndarray[DTYPE_t] deltaLossY, np.ndarray[DTYPE_t] y):
    deltaLossX = deltaLossY * (1 - y) * y