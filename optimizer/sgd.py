import numpy as np

class SGD:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def update(self, weight, lossWeight):
        weight -= lossWeight * self.learningRate