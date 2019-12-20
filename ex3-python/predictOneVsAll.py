import numpy as np
from scipy.special import expit

def predictOneVsAll(all_theta, X):
    m,n = X.shape
    k = all_theta.shape[0]
    theta = all_theta.reshape(k, n+1)
    X = np.column_stack((np.ones((m,1)), X))
    p = np.argmax(expit( np.dot(X,theta.T) ), axis=1)+1
    return p.reshape(m,1)
