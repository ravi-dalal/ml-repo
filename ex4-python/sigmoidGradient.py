import numpy as np
from scipy.special import expit

def sigmoidGradient (z):
    sigGrad = np.multiply (expit(z), (1.0 - expit(z)))
    return sigGrad
