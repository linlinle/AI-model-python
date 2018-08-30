"""
在iris数据集上绘制决策树的决策表面
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

n_class = 3
plot_colors = "bry"
plot_step = 0.02

iris = load_iris()

for pairidx, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):

    X = iris.data[:,pair]
    y = iris.target

    clf = DecisionTreeClassifier()
    clf.fit(X,y)

    plt.subplot(2,3,pairidx +1)

    x_min, x_max = X[:,0].min()-1 , X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1 , X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    for i, color in zip(range(n_class), plot_colors):
        idx = np.where(y==i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label = iris.target_names[i],cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()