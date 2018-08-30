# -*- coding: utf-8 -*-
'''softmax'''
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])



probabily = softmax(scores)


 # Plot softmax curves
plt.plot(x, probabily.T, linewidth=2)
plt.show()

