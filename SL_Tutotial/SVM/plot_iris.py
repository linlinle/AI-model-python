# -*- coding: utf-8 -*-
'''
Comparison of different linear SVM classifiers on a 2D projection of the iris dataset.
We only consider the first 2 features of this dataset:Sepal length、Sepal width

This example shows how to plot the decision surface for four SVM classifiers with different kernels.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

def make_meshgrid(x,y,h=.02):
    """创建要绘制的点的网格
    Parameters
    ----------
    x: 数据以x轴网格为基础
    y: data to base y-axis meshgrid on
    h: meshgrid步长，可选

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    #   #接受两个数组一维数组，产生两个二维矩阵
    xx, yy = np.meshgrid(np.arange(x_min,x_max),
                         np.arange(y_min,y_max))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
    # np.c_是按行连接两个矩阵;ravel()将多维数组降为一维
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    # contour和contourf都是画三维等高线图的，不同点在于contourf会对等高线间的区域进行填充
    out = ax.contourf(xx, yy, Z, **params)
    return out


#   import data
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:,:2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # 正则化参数
models= (svm.SVC(kernel='linear',C = C),
         svm.LinearSVC(C=C),
         svm.SVC(kernel='rbf',gamma=0.7,C=C),
         svm.SVC(kernel='poly',degree=3,C=C))
models = (clf.fit(X,y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# 设置2*2的网格画板,4个坐标图
fig,sub = plt.subplots(2,2)
#plt.subplots_adjust(wspace=4,hspace=0.4)

# 将2个特征分开，坐标
X0 , X1 = X[:,0] , X[:,1]
xx, yy = make_meshgrid(X0,X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm,alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()