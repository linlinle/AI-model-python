"""GaussianNB 实现了运用于分类的高斯朴素贝叶斯算法。特征的可能性(即概率)假设为高斯分布:
参数 \sigma_{y} 和 \mu_{y} 使用最大似然法估计。


"""
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB   # 可替换分类器


iris= load_iris()
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

print("Number of mislabeled points out of a total %d points : %d" %(iris.data.shape[0],(iris.target != y_pred).sum()))