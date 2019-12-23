import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import polyFeatures

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
