# -*- coding: utf-8 -*-
'''
    噪声（非信息性的）特征被添加到iris data，并应用单变量特征选择。 对于每个特征，我们绘制单变量特征选择的p值和SVM的相应权重。
我们可以看到，单变量特征选择选择了信息特征而不是噪声，并且这些信息特征具有较大的SVM权重。
    在所有特征中，只有前四个特征是重要的。 通过单变量特征选择，我们可以看到他们的得分最高。在SVM之前应用单变量特征选择可以改善分类。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

# #############################################################################
# Import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(1,0.1,size=(len(iris.data),20))   #均匀分布
# Add the noisy data to the informative features
X = np.hstack((iris.data,E))
#X = np.c_[iris.data,E]
y = iris.target

plt.figure()
plt.clf() # Clear the current figure.

X_indices = np.arange(X.shape[-1])

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
# f_classif: Compute the ANOVA F-value for the provided sample.
selector = SelectPercentile(f_classif, percentile=10) #SelectPercentile 选择排名排在前n%的变量 ;percentile(Percent of features to keep)
# fit后获得pvalues_和scores_属性。他们shape为[1,n_feature]
selector.fit(X,y) # Run score function on (X, y) and get the appropriate features.
scores = -np.log10(selector.pvalues_)
scores /= scores.max() # 0~1归一化
plt.bar(X_indices - 0.45,scores,width=.2,
        label='Univariate score ($-Log(p_{value})$)',
        color='darkorange',
        edgecolor='black')

# #############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X,y)
#   coef_：[3,24]
svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
        color='navy', edgecolor='black')

clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X),y) # Reduce X to the selected features
#   coef_：[3,3]
svm_weights_selector = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selector /= svm_weights_selector.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selector,width=.2,
        label='SVM weights after selection',color='c',edgecolor='black')

plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(()) #    去除y坐标
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()