# -*- coding: utf-8 -*-
'''
SVM with custom kernel
Simple usage of Support Vector Machines to classify a sample. It will plot the decision surface and the support vectors.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

def my_kernel(X, Y):
    """
        We create a custom kernel:

                     (2  0)
        k(X, Y) = X  (    ) Y.T
                     (0  1)
    """
    M = np.array([[2,0],[0,1.0]])
    return np.dot(np.dot(X,M),Y.T)

iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

h = 0.2 # 网格中的步长

clf = svm.SVC(kernel=my_kernel)
clf.fit(X,Y)

x_min, x_max = X[:,0].min() - 1,X[:,0].max() +1
y_min, y_max = X[:,1].min() - 1,X[:,1].max() +1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()