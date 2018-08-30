
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    dataMat ,labelList= [],[]
    with open("testSet.txt") as f:
        for line in f.readlines():
            lineList =line.strip().split()
            dataMat.append([1.0, float(lineList[0]), float(lineList[1])])
            labelList.append(int(lineList[2]))
    return dataMat,labelList

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def gradient_ascent(dataMat, classList):
    """
    梯度上升与梯度下降类似，一个求最大值，一个求最小值
    :param dataMat:
    :param classList:
    :return:
    """
    dataMarix = np.mat(dataMat)
    labelList = np.mat(classList).transpose()
    m,n = np.shape(dataMarix)
    alpha = 0.01
    weights = np.ones((n,1))
    for i in range(500):
        predict = sigmoid(dataMarix*weights)            #np.mat不需要求和
        error = labelList - predict
        weights = weights + alpha*dataMarix.transpose()*error

    return weights

def stochastic_gradient_ascent(dataMat, classList, numIter=150):
    dataMarix = np.array(dataMat)
    labelList = np.array(classList).transpose()
    m,n = np.shape(dataMarix)
    weights = np.ones(n)
    for j in range(numIter):
        dataInex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataInex)))
            predict = sigmoid(sum(dataMarix[randIndex]*weights))    #np,array需要求和
            error = labelList[randIndex]-predict
            weights = weights - alpha*error*dataMarix[randIndex]
            del (dataInex[randIndex])
    return weights


def plotBestFit(weights):
    dataMat, labelMat = load_dataset()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1,ycord1,xcord2,ycord2 = [],[],[],[]
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y.transpose())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ =="__main__":
    dataArr, labelMat = load_dataset()
    weights = gradient_ascent(dataArr, labelMat)
    plotBestFit(weights)