import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

def visualizeBoundaryLinear(X, y, model):
    #print("coef=", model.coef_)
    #print("intercept=", model.intercept_)
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]
    #print('xp= ', xp)
    #print('yp=' , yp)

    plt = plotData(X, y)
    plt.plot(xp, yp, 'b-')
    plt.show()