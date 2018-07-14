import numpy as np


def ReLu(x):
    return max(0, x)


def applyFilter(original, filter, i, j):
    sX = filter.shape[0]
    sY = filter.shape[1]
    return np.sum(original[i-1:i+sX-1, j-1:j+sY-1] * filter)


np.random.seed(1)

X = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

Y = np.array([
    [-2, 0, 2, 2, 0, 0],
    [-2, 0, 1, 1, 0, 1],
    [-1, 0, -1, -1, 0, 2],
    [0, 0, -2, -2, 0, 2],
])

filter = np.random.random((3, 3))


row = 4
col = 6
fil = 3
l1Row = row-fil+1
l1Col = col-fil+1
border = (fil-1)//2

print(filter)
for it in range(6000):

    result = np.empty([l1Row, l1Col])

    for i in range(border, row-border):
        for j in range(border, col-border):
            result[i-1, j-1] = applyFilter(X, filter, i, j)

    error = Y-result

    delta = np.empty([fil, fil])
    for i in range(fil):
        for j in range(fil):
            delta[i, j] = np.sum(X[i:i+l1Row, j:j+l1Col] * error)

    if it % 1000 == 0:
        print("Filter", np.around(filter, decimals=1))
        print("Result", np.around(result, decimals=1))
        print("Error", np.around(error, decimals=1))
    # print(delta * 0.01)
    filter += delta * 0.01

print(filter)
