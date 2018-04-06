from math import sqrt
import numpy as np

X = np.array([0, 1, 2, 3])
Y = np.array([0, 1, 2, 4])

def dist (X, Y):
    sum = 0
    Z = X - Y

    if X.shape != Y.shape:
        raise ValueError('Vectors must be contain the same dimensions')

    for i in range(X.shape[0]):
        sum += pow(Z[i], 2)

    return sqrt(sum)


print(dist(X, Y))
