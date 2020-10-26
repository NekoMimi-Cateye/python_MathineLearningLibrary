import numpy as np
import Model as mdl
from layers import fullconnect as fc
from activations import sigmoid as sig
from lossfunctions import mse as mse
from optimizer import sgd as sgd

if __name__ == "__main__":
    lr = 0.1
    ep = 10000
    model = mdl.Model()
    model.addLayer(fc.FullConnect(2, 6), True)
    model.addLayer(sig.Sigmoid(6), False)
    model.addLayer(fc.FullConnect(6, 2), True)
    model.addLayer(sig.Sigmoid(2), False)
    model.addLossFunction(mse.MSE(2))
    model.addOptimizer(sgd.SGD(lr))

    x = np.array([[0, 1]], dtype = float)
    t = np.array([[0, 1]], dtype = float)
    
    y = model.predict(x)
    print(y)

    model.train(x, t, ep)

    y = model.predict(x)
    print(y)