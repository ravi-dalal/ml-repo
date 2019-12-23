import numpy as np
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg

def learningCurve (x, y, xval, yval, _lambda):
    m, n = x.shape
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    for i in range(0, m):
        x1 = x[0:i+1,:]
        y1 = y[0:i+1,:]
        result = trainLinearReg (x1, y1, _lambda)
        theta = result.x
        error_train[i] = linearRegCostFunction(theta, x1, y1, _lambda)
        error_val[i] = linearRegCostFunction(theta, xval, yval, _lambda)
    return error_train, error_val
