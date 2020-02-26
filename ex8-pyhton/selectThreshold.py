import numpy as np

def selectThreshold(yval, pval):

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    epsilons = np.linspace(min(pval), max(pval), 1000)
    pval = pval.reshape(pval.size, 1)
    for epsilon in epsilons:
        #print('epsilon = ', epsilon)
        tp = sum(np.logical_and((pval < epsilon), (yval == 1)))
        fp = sum(np.logical_and((pval >= epsilon), (yval == 1)))
        fn = sum(np.logical_and((pval < epsilon), (yval == 0)))
        if ((tp + fp) == 0 or (tp + fn) == 0 or tp == 0):
            F1 = [0]
        else: 
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2*precision*recall / (precision + recall)
        #print('F1 = ', F1)
        if (F1[0] > bestF1):
            bestF1 = F1[0]
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1
        
        

