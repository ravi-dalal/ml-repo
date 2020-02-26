import numpy as np
import scipy.linalg as linalg

def multivariateGaussian(X, mu, sigma2):
    
    k = len(mu)
    
    # turns 1D array into 2D array
    #if sigma2.ndim == 1:
    #    sigma2 = np.reshape(sigma2, (-1,sigma2.shape[0]))

    if (sigma2.shape[1] == 1 or sigma2.shape[0] == 1):
        sigma2 = linalg.diagsvd(sigma2.flatten(), len(sigma2.flatten()), len(sigma2.flatten()))

    #X = X - mu.reshape(mu.size, order='F')
    X = X - mu.flatten(order='F')
    #print("X shape=", X.shape)
    #print("sigma2 shape=", sigma2.shape)

    p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma2), -0.5) ) * \
        np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))

    return p