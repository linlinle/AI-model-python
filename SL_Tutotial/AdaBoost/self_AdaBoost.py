
"""基于单层决策树构建AdaBoost分类器"""
import numpy as np
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过阈值比较对数据进行分类的，阈值一边分为-1，另一边为+1。通过数组过滤实现
    :param dataMatrix:
    :param dimen: 特征维度的索引
    :param threshVal: 阈值
    :param threshIneq: < 或者 其他
    :return:
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))            #[n_samples,1]
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    遍历stumpClassify函数所有的可能输入值，并找到数据集上最佳的单层决策树，这里"最佳"是基于数据的权重D来定义
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    流程：
    将最小误差率 minError设为无穷
    For every feature in the dataset(第一层循环):
        对每个步长（第二层循环）:
            对每个不等号（第三层循环）:
                建立一颗单层决策树并利用加权数据集对它进行测试
                如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    Return the best stump
    """
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0                                         # 用于在特征的所有可能值上进行遍历
    bestStump = {}                                          # 存储给定权重向量D时所得到的最佳单层决策树的weights
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf                                          # 初始化为无穷大，之后寻找最小错误率
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    对每次迭代：
        利用buildStump()函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树数组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率为0。0，则退出循环
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # init D to all equal
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        # print "D:",D.T
        alpha = float(
            0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = np.multiply(D, np.exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    利用训练得到的多个弱分类器进行分类
    :param datToClass:
    :param classifierArr:
    :return:
    """
    dataMatrix = np.mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas);
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)



if __name__ == "__main__":
    X, y = make_hastie_10_2(n_samples=12000, random_state=1)
    # X_test, y_test = X[2000:], y[2000:]
    # X_train, y_train = X[:2000], y[:2000]
    # datMat, classLabels = loadDataSet("horseColicTraining2.txt")
    # classifierArray = adaBoostTrainDS(X_train, y_train, 20)
    # testArr, testLabelArr = loadDataSet("horseColicTest2.txt")
    # prediction10 = adaClassify(X_test, classifierArray[0])
    # a = (prediction10 != np.mat(y_test).T)
    # errArr = np.mat(np.ones((10000, 1)))
    # print(errArr[prediction10!=np.mat(y_test).T].sum())

    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(X, y, 10)
    plotROC(aggClassEst.T,y)
