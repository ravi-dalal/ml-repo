import numpy as np

def mapFeature(X1, X2):
    r = X1.shape[0]
    degree = 6
    out = np.ones((r,1))
    for i in range(1,degree+1):
        for j in range (0,i+1):
            newcol = (np.multiply(np.power(X1,(i-j)),np.power(X2,j)))
            out = np.hstack((out, newcol))
    return out
