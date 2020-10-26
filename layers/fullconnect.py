import numpy as np
class FullConnect:
    def __init__(self, inLen, outLen):
        self.dataX = np.zeros((1, inLen))
        self.weight = np.zeros((inLen, outLen))
        self.bias = np.zeros((1, outLen))
        self.deltaLossW = np.zeros((inLen, outLen))
        self.deltaLossB = np.zeros((1, outLen))

    def forward(self, data):
        self.dataX = data
        return data@self.weight + self.bias

    def backward(self, loss):
        deltaLoss = loss@self.weight.T
        self.deltaLossW = self.dataX.T@loss
        self.deltaLossB = loss
        return deltaLoss