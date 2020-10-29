import numpy as np
import Model as mdl
from layers import fullconnect as fc
from activations import Sigmoid as sig
from lossfunctions import MSE as mse
from optimizer import sgd as sgd

def a():
    print("a")

if __name__ == "__main__":
    A = a
    a()
    print(type(A))
    lr = 0.1
    ep = 1000000
    model = mdl.Model()
    model.addLayer(fc.FullConnect(2, 6), True)
    model.addLayer(sig.Sigmoid((1, 6)), False)
    model.addLayer(fc.FullConnect(6, 2), True)
    model.addLayer(sig.Sigmoid((1, 2)), False)
    model.addLossFunction(mse.MSE(2))
    model.addOptimizer(sgd.SGD(lr))

    x = np.array([[0, 1]], dtype = float)
    t = np.array([[1, 0]], dtype = float)
    
    y = model.predict(x)
    print(y)

    model.train(x, t, ep)

    y = model.predict(x)
    print(y)