import numpy as np

def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((n, 1))

    for i in range(n):
        x[i] = 1 if i in word_indices else 0

    return x
