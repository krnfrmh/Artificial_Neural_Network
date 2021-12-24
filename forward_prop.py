import numpy as np
import matplotlib.pyplot as plt


def get_data(Nclass):
  X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
  X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
  X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
  
  X = np.vstack([X1, X2, X3])
  Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
  return X, Y

def sigmoid(a):
  return 1 / (1 + np.exp(-a))

def forward(X, W1, b1, W2, b2):
    # sigmoid hidden layer
    Z = sigmoid(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    # softmax
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y
    
# determine the classification rate
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    # num correct / num total
    return float(n_correct) / n_total

X, Y = get_data(Nclass)

D = 2 # Input dimensionality
M = 3 # Hidden layer size
K = 3 # Number of classes

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

P_Y_given_X = forward(X, W1, b1, W2, b2)
# Taking argmax over softmax probabilities
P = np.argmax(P_Y_given_X, axis=1)
