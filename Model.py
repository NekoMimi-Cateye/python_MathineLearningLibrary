import numpy

class Model:
    def __init__(self):
        self.model = []
        self.lossFunction = []
        self.wlayerPointer = []
        self.optimizer = []
        self.logLoss = []
        self.logEpoch = []

    def addLayer(self, layer, haveWeight):
        self.model.append(layer)
        if haveWeight:
            self.wlayerPointer.append(len(self.model)-1)

    def addLossFunction(self, lossFunction):
        self.lossFunction.append(lossFunction)

    def addOptimizer(self, optimizer):
        self.optimizer.append(optimizer)

    def predict(self, x):
        y = self.model[0].forward(x)
        for m in self.model[1:]:
            y = m.forward(y)
        return y

    def backward(self):
        deltaLoss = self.lossFunction[0].backward()
        for m in self.model[len(self.model)-1:0:-1]:
            deltaLoss = m.backward(deltaLoss)
        


    def train(self, dataX, dataT, epochs):
        self.logLoss = [0.0 for i in range(epochs)]
        self.logEpoch = [i+1 for i in range(epochs)]
        for e in range(epochs):
            ##--step1--##
            dataY = self.predict(dataX)

            ##--step2--## #in production
            loss = self.lossFunction[0].forward(dataY, dataT)
            
            ##--step3--##
            self.backward()

            ##--step4--##
            for i in self.wlayerPointer:
                self.optimizer[0].update(self.model[i].weight, self.model[i].deltaLossW)
                self.optimizer[0].update(self.model[i].bias, self.model[i].deltaLossB)

            ##--step5--##
            self.logLoss[e] += loss