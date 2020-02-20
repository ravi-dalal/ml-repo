import numpy as np

def estimateGaussian(X):

    mu = np.mean(X, axis=0)
    mu = mu.reshape(mu.size, 1)
    sigma2 = np.var(X, axis=0)
    sigma2 = sigma2.reshape(sigma2.size, 1)
    #print("mu shape=", mu.shape)
    #print("sigma2 shape=", sigma2.shape)
    return mu.T, sigma2.T
    