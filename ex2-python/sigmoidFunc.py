import numpy as np

def sigmoidFunc(x):
    return np.divide(1.0, (1.0 + np.exp(-x)))
