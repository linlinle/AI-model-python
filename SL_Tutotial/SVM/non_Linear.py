# -*- coding: utf-8 -*-
'''
Perform binary classification using non-linear SVC with RBF kernel. The target to predict is a XOR of the inputs.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3,3,500),
                     np.linspace(-3,3,500))
np.random.seed(0)# Seed the generator,This method is called when RandomState is initialized

X = np.random.randn(300,2)
Y = np.logical_xor(X[:,0]>0,X[:,1]>0)

clf = svm.NuSVC()
clf.fit(X,Y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()]) #对每个网格分类
Z = Z.reshape(xx.shape)
#Display an image on the axes，将整个区域按照数值大小配色
plt.imshow(Z,interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()