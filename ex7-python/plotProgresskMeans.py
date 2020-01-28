import matplotlib.pyplot as plt
import numpy as np
from plotDataPoints import plotDataPoints
from drawLine import drawLine

def plotProgresskMeans(X, centroids, previous, idx, K, i):
	# Plot the examples
	plotDataPoints(X, idx, K)
	# Plot the centroids as black x's
	plt.plot(centroids[:,0], centroids[:,1], marker='x', markeredgecolor='k', markersize=10, linewidth=3, linestyle='None')
	#plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=400, c='k', linewidth=1)

	# Plot the history of the centroids with lines
	for j in range (centroids.shape[0]):
	    drawLine(centroids[j, :], previous[j, :], c='b')

	# Title
	plt.title('Iteration number '+str(i))
	#return plt