import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier

def plotROC(predStrengths, classLabels):
    """
    每遇到一个+1标签，沿着y轴下降一个步长，降低真正例率；
    每遇到一个其他标签，沿着x轴倒退一个步长，降低假正例率；
    :param predStrengths:
    :param classLabels:
    :return:
    """
    cursor = (1.0, 1.0)                                 # 游标位置
    ySum = 0.0                                          # 计算AUC的变量
    numPositiveClass = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPositiveClass)                 # 确定了y轴步长
    xStep = 1 / float(len(classLabels) - numPositiveClass)# 确定了y轴步长

    #数组值从小到大的索引值
    sortedIndicies = predStrengths.argsort()             #从小到大顺序排列，从(1.0，1.0)开始画一直到(0,0)
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist():
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cursor[1]
        # draw line from cursor to (cursor[0]-delX,cursor[1]-delY)
        ax.plot([cursor[0], cursor[0] - delX], [cursor[1], cursor[1] - delY], c='b')
        cursor = (cursor[0] - delX, cursor[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC cursorve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    # 每个小矩形相加，矩形的宽度为xStep，因此对矩形的高度进行相加得到ySum
    print("the Area Under the cursorve is: ", ySum * xStep)

if __name__ == "__main__":
    X, y = make_hastie_10_2(n_samples=4000, random_state=1)
    X_test, y_test = X[2000:], y[2000:]
    X_train, y_train = X[:2000], y[:2000]
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    preds = clf.predict_proba(X_test)
    plotROC(preds[:,1],y_test)
