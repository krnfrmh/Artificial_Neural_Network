import numpy as np
import matplotlib.pyplot as plt


def get_data(Nclass):
  X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
  X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
  X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
  
  X = np.vstack([X1, X2, X3])
  Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
  return X, Y

X, Y = get_data(Nclass)

D = 2 # Input dimensionality
M = 3 # Hidden layer size
K = 3 # Number of classes

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)
