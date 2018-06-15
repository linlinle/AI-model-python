# -*- coding: utf-8 -*-
'''
Plot the maximum margin separating hyperplane within a two-class separable dataset using a Support Vector Machine classifier with linear kernel.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
X, y = make_blobs(n_samples=40,centers=2,random_state=6)

# fit the Re_classifying, 出于说明目的，这里不需要正则化
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(X, y)

# 画出所有样本点
plt.scatter(X[:,0],X[:,1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca() # Get Current Axes
xlim = ax.get_xlim() # x轴区间
ylim = ax.get_ylim() # y轴区间

# create grid to evaluate Re_classifying
xx = np.linspace(xlim[0], xlim[1], 30) # np.arange升级版，x轴区间分为30个点
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx) # 从坐标向量返回坐标矩阵
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, color='black', levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
# plot support vectors
print(clf.support_vectors_)
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,linewidths=1,facecolor='black')


plt.show()
