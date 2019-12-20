import numpy as np
from scipy.special import expit

def predict (theta1, theta2, X, y):
    #print(X)
    m, n = X.shape
    a1 = np.column_stack((np.ones((m,1)), X))   
    a2 = expit(np.dot(a1, theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
#    print(a2.shape)
    a3 = expit(np.dot(a2, theta2.T))
    #print(a3.shape)
    #p = np.max(a3, axis=1)
    #p = p.reshape(m,1)
    #print(a3)
    pindex = np.argmax(a3, axis=1)+1
    pindex = pindex.reshape(m,1)
    #print(pindex)
    return pindex
