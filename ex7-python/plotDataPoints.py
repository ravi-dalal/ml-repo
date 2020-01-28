import matplotlib.pyplot as plt
#from matplotlib.colors import hsv_to_rgb
from hsv import hsv
import numpy as np

def plotDataPoints(X, idx, K):

	# Create palette
	palette = hsv(K + 1)
	#print('palette = ',palette)
	colors = np.array(palette[idx.flatten().astype(int), :])
	#colors = np.array([palette[int(i)] for i in idx]) 
	
	# Plot the data
	#plt.scatter(X[:,0], X[:,1], s=15, edgecolors=colors)
	plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=colors)
