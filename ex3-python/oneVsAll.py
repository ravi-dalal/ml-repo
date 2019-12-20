import numpy as np
from lrCostFunction import lrCostFunction
from scipy.optimize import minimize

def oneVsAll(X, y, num_labels, _lambda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.column_stack((np.ones((m,1)), X))    
    for c in range(1, num_labels+1):
        #print(y==c)
        initial_theta = np.zeros((n + 1, 1)).flatten()
        print("Training {:d} out of {:d} categories...".format(c, num_labels))
        fargs = (X, (y==c).reshape(m,1), _lambda)
        theta = minimize(lrCostFunction, x0=initial_theta, args=fargs, method="CG" \
                         , options={'maxiter':50}, tol=0.005)
        #print(theta)
        all_theta[c-1,:] = theta["x"]
    return all_theta
