import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

def visualizeBoundary(X, y, model):
    plt = plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    
    for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(this_X)
    
    #print(vals)
    # Plot the SVM boundary
    CS = plt.contour(X1, X2, vals)
    print("Levels = ", CS.levels)
    plt.clabel(CS, fmt = '%.2f', colors = 'k', fontsize=10)
    plt.show()