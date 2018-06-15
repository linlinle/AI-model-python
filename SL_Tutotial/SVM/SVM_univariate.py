# -*- coding: utf-8 -*-
'''
SVM-Anova: SVM with univariate feature selection¶

单变量特征选择能够对每一个特征进行测试，衡量该特征和响应变量之间的关系，根据得分扔掉不好的特征。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets,feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# #############################################################################
# Import some data to play with
digits = datasets.load_digits()
y = digits.target
# 丢弃数据，用以处于维数灾难中
y = y[:200]
X = digits.data[:200]
n_samples = len(y)
x = X.reshape((n_samples,-1)) # 先满足行是n_samples条件
# add 200 non-informative features
X = np.hstack((X,2*np.random.random((n_samples,200))))

# #############################################################################
# Create a feature-selection transform and an instance of SVM that we combine together to have an full-blown estimator
#创建一个特征选择变换和一个SVM实例，我们将它们组合在一起以形成一个完整的估计器
transform = feature_selection.SelectPercentile(feature_selection.f_classif)
clf = Pipeline([('anova',transform),('svc',svm.SVC(C=1.0))])

# #############################################################################
# Plot the cross-validation score as a function of percentile of features
# 将交叉验证分数绘制为特征百分位数的函数
score_means = list()
score_stds = list()
percentiles = (1,3,6,10,15,20,30,40,60,80,100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using 1 CPU
    this_scores = cross_val_score(clf,X,y,n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles,score_means, np.array(score_stds))
plt.title('Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')
plt.axis('tight')
plt.show()