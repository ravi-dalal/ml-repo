import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature

def plotDecisionBoundary (theta, X, y):
    plt, pos, neg = plotData(X[:,1:3], y)
    r,c = X.shape
    if c <= 3:
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        db = plt.plot(plot_x, plot_y)[0]
        plt.legend((pos, neg, db), ('Admitted', 'Not Admitted', 'Decision Boundary'), loc='lower left')
        plt.axis([30, 100, 30, 100])
    else:
	    u = np.linspace(-1, 1.5, 50)
	    v = np.linspace(-1, 1.5, 50)
	    z = np.zeros((len(u), len(v)))
	    for i in range(1,len(u)):
	        for j in range(1,len(v)):
	            z[i,j] = np.dot(mapFeature(np.asarray([[u[i]]]), np.asarray([[v[j]]])),theta)
	    z = z.T
	    #print(z)
	    CS = plt.contour(u, v, z, levels=[0])
	    db = CS.collections[0]
	    plt.clabel(CS, fmt = '%2.1d', colors = 'g', fontsize=14)
	    plt.legend((pos, neg, db),('y = 1', 'y = 0', 'Decision Boundary'), loc='upper right')

    return plt