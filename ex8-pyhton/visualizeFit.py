import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariateGaussian

def visualizeFit(X, mu, sigma2):

    X1, X2 = np.meshgrid(np.arange(0, 35.1, 0.5), np.arange(0, 35.1, 0.5))
    Z = multivariateGaussian(np.column_stack((X1.reshape(X1.size, order='F'), X2.reshape(X2.size, order='F'))), mu, sigma2)
    Z = Z.reshape(X1.shape, order='F')

    plt.plot(X[:, 0], X[:, 1],'bx')
    # Do not plot if there are infinities
    if (np.sum(np.isinf(Z)) == 0):
        plt.contour(X1, X2, Z, np.power(10,(np.arange(-20, 0.1, 3)).T))
    