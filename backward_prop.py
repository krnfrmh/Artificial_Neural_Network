import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    logit = X.dot(W1) + b1
    # 1st Hidden Layer
    Z = 1 / (1 + np.exp(-1 * logit))
    # Output Layer
    A = Z.dot(W2) + b2
