import matplotlib.pyplot as plt

def plotData(x, y):
    plt.scatter(x, y, marker='x', label='Training Data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(loc='lower right')
    return plt