# -*- coding: utf-8 -*-
'''softmax'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     row,column = x.shape[0],x.shape[1]
#     for i in range (0,column):
#         sum_scores = sum(x[:,i])
#         for j in range(0,row):
#             x[j,i] = x[j,i]/sum_scores
#     return x

 # Plot softmax curves
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

