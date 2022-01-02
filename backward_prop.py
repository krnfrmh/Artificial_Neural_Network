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
