import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.io import loadmat

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
    plt.show(block=False)
    return h, display_array

def predict (theta1, theta2, X, y):
    #print(X)
    m, n = X.shape
    a1 = np.column_stack((np.ones((m,1)), X))   
    a2 = expit(np.dot(a1, theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
#    print(a2.shape)
    a3 = expit(np.dot(a2, theta2.T))
    #print(a3.shape)
    #p = np.max(a3, axis=1)
    #p = p.reshape(m,1)
    #print(a3)
    pindex = np.argmax(a3, axis=1)+1
    pindex = pindex.reshape(m,1)
    #print(pindex)
    return pindex
    
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)
data = loadmat("D:\Learn\ML\machine-learning-ex3\ex3\ex3Data1.mat")
X = data['X']
y = data['y']
#print(y)
weights = loadmat("D:\Learn\ML\machine-learning-ex3\ex3\ex3weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']
#print(X.shape)
#print(theta1.shape)
#print(theta2.shape)
#pred = predict(theta1, theta2, X, y)
#print(pred.shape)
#print("Train Accuracy: ",np.mean(pred == y)*100)

rand_indices = np.random.permutation(X.shape[0])
r = rand_indices[0:3]
#print(r)
sel = np.take(X, r, 0)
print(sel)
print(np.take(y,r, 0))
displayData(sel)
pred = predict(theta1, theta2, sel, y)
print(pred)

#for x in range(1, 10):
    