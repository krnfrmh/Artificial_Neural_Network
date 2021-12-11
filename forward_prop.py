import numpy as np
import matplotlib.pyplot as plt


def get_data(Nclass):
  X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
  X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
  X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
  
  X = np.vstack([X1, X2, X3])
  Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
  return X, Y
