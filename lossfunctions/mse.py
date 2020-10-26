import numpy as np
class MSE:
    def __init__(self, shape):
        self.dataY = np.empty(shape)
        self.dataT = np.empty(shape)

    def forward(self, dataY, dataT):
        self.dataY = dataY
        self.dataT = dataT
        return np.average(0.5 * (dataY - dataT) * (dataY - dataT))

    def backward(self):
        return self.dataY - self.dataT