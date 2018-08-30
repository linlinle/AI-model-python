"""
KNeighborsClassifier 基于每个查询点的 k 个最近邻实现，
默认值 weights = 'uniform' 为每个近邻分配统一的权重。而 weights = 'distance' 分配权重与查询点的距离成反比。 或者，用户可以自定义一个距离函数用来计算权重。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

n_neighbors = 15

iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target

h = .02

#   creat color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_blod = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weight in ["uniform","distance"]:
    clf = neighbors.KNeighborsClassifier(n_neighbors=25,weights=weight)
    clf.fit(X,y)

    x_min, x_max = X[:,0].min() -1, X[:, 0].max()+1
    y_min, y_max = X[:,1].min()-1,X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),
                         np.arange(y_min,y_max,h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  #np.hstack()

    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap = cmap_light)

    plt.scatter(X[:,0],X[:,1], c=y, cmap=cmap_blod,edgecolors='k',s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k=%s, weight=%s)"%(n_neighbors,weight))

plt.show()