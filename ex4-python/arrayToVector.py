import numpy as np

def arrayToVector (a, b):
    return np.concatenate((a, b),axis=None)
    #return np.concatenate((a.reshape(a.size, order='F'), b.reshape(b.size, order='F')))
    #return np.concatenate((np.ravel(a, order='F'), np.ravel(b, order='F')))
