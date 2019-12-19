import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    y1 = np.where(y == 1)
    y2 = np.where(y == 0)
    pos = plt.plot(X[y1,0], X[y1,1], marker='+', color='k', linestyle='None')[0]
    neg = plt.plot(X[y2,0], X[y2,1], marker='.', color='y', linestyle='None')[0]
    return plt, pos, neg
