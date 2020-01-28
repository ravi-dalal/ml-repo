import numpy as np
from matplotlib import colors

def hsv(n):
    return colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, n), np.ones(((n), 2)) ]) )