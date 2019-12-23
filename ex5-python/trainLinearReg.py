import numpy as np
from linearRegCostFunction import linearRegCostFunction
from scipy.optimize import minimize

def trainLinearReg (x, y , _lambda):
    m, n = x.shape
    fargs = (x, y, _lambda)
    initial_theta = np.zeros((x.shape[1], 1))
    return minimize(linearRegCostFunction, x0=initial_theta, args=fargs, method="CG", \
    		 options={'disp': False, 'maxiter': 200.0})
