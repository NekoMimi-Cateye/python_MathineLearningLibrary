import cython
import numpy as np
cimport numpy as np

def void SGD(self, np.ndarray data, np.ndarray lossData, float learningRate):
    data -= lossData * learningRate