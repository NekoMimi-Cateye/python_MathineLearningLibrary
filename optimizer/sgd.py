import numpy as np

class SGD:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def sgd(self, weight, lossWeight):
        weight -= weight * self.learningRate