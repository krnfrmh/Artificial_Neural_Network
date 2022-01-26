import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    logit = X.dot(W1) + b1
    # 1st Hidden Layer
    Z = 1 / (1 + np.exp(-1 * logit))
    # Output Layer
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    # softmax
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

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

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    # Calculate gradient
    ret2 = Z.T.dot(T - Y)
    return ret2

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape
    # Calculate gradient
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret1 = X.T.dot(dZ)
    return ret1
    
def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
    
def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()
    
    
