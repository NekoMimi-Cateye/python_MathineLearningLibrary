import numpy as np
class Sigmoid:
    def __init__(self, shape):
        self.dataY = np.empty(shape)

    def forward(self, data):
        self.dataY = 1 / (1 + np.exp(-data))
        return self.dataY

    def backward(self, loss):
        deltaLoss = self.dataY * (1 - self.dataY) * loss
        return deltaLoss