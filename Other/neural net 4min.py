import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

for j in range(100000):

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2

    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    if j == 100:
        # print(l0)
        # print ("dot product", np.dot(l0, syn0))
        # print("l1", l1)
        # print ("dot product", np.dot(l1, syn1))
        # print(l2, nonlin(l2, deriv=True))
        print("syn1Before", syn1)
        print("l1", l1, np.sign(l1))
        print("l1", l1.T)
        print("increment", l1.T.dot(l2_delta))
        print("newIncrement", np.sign(l1.T).dot(l2_delta))
        print("delta", l2_error * nonlin(l2, deriv=True))
        print("syn1After", syn1 + l1.T.dot(l2_delta))
        print("Error:" + str(np.mean(np.abs(l2_error))))

    syn1 += np.sign(l1.T).dot(l2_delta)
    syn0 += np.sign(l0.T).dot(l1_delta)


print("Output after training")
print(l2)
