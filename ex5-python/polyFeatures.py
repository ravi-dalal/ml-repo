import numpy as np

def polyFeatures(x, p):
    x_poly = x
    for i in range (1, p):
        x_poly = np.column_stack((x_poly, np.power(x,i+1)))
    return x_poly
