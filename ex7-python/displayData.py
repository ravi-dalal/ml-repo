import numpy as np
import math
import matplotlib.pyplot as plt

def displayData (X):
    example_width = int(round(math.sqrt(np.size(X,1))))
    plt.gray()
    m,n = X.shape
    example_height = int(n / example_width)
    
    display_rows = int(math.floor(np.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    pad = 1;
    
    display_array = np.ones((pad + display_rows * (example_height + pad), \
                             pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for j in range(0, display_rows):
        for i in range (0, display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(X[curr_ex, :]))
            rows = pad + j * (example_height + pad) + np.array(range(example_height))
            cols = pad + i * (example_width + pad) + np.array(range(example_width))
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = \
            np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") \
            / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    # Do not show axis
    plt.axis('off')
    return plt, h, display_array
