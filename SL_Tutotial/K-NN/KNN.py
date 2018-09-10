
import os
import numpy as np
import matplotlib.pyplot as plt

def knnClassifier(X, dataset, labels, k):
    """

    :param X: 待预测样本(1*M)
    :param dataset: 数据集(N*M)
    :param labels: (N*1)
    :param k: 距离最近的k个点
    :return:  出现次数最对的类
    """
    datasetSize = dataset.shape[0]                   # 样本数量

    #  计算与其他样本距离
    diffMat = np.tile(X,(datasetSize,1)) - dataset   # 数据扩充
    squrediffMat = diffMat**2                       # 欧式距离
    squreDistances = squrediffMat.sum(axis=1)       # 距离和
    distances = squreDistances**0.5                 # 平方根

    #   由小到大
    sortedDistance = sorted(list(zip(distances,labels))) #将所有样本与X的距离进行排序

    # 选择距离最小的k个点
    classCount = dict()
    for i in range(k):
        voteLabel = sortedDistance[i][1]
        classCount[voteLabel] = classCount.get(voteLabel,0) +1  # 相同类别累计加一
    sortedclassCount = sorted(classCount.items(), key=lambda x:x[1],reverse=True)   #不同类别出现次数，由大到小排序
    return sortedclassCount[0][0]


############################################   简单数据集      ##########################################################

def creatSimpleDatasets():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A',"B","B"]
    return group, labels
def testSimpleDatasets():
    dataSets, labels = creatSimpleDatasets()
    print(knnClassifier([0.2,0.2],dataSets,labels,3))

##########################################      约会网站配对效果    #####################################################

def fileToArray(file_name):
    with open(file_name) as f:
        Lines = f.readlines()
    returnArray = np.zeros((len(Lines),3))
    labels = []
    for index,line in enumerate(Lines):
        lineList = line.strip().split("\t")
        returnArray[index,:] = lineList[:3]
        labels.append(int(lineList[-1]))
    return np.array(returnArray),labels
def normalization(dataset):
    minVal = dataset.min(0)#np.min(dataset,axis=0)
    maxVal = np.max(dataset,axis=0)
    ranges = maxVal-minVal
    N = dataset.shape[0]

    #   归一化公式
    normDataSet = dataset - np.tile(minVal,(N,1))
    normDataSet = normDataSet/np.tile(ranges,(N,1))
    return normDataSet
def testDating():
    dataSets, labels = fileToArray("datingTestSet2.txt")
    normData = normalization(dataSets)
    N = normData.shape[0]
    errorCount = 0
    for i in range(N):
        result = knnClassifier(normData[i,:],normData,labels,3)
        print("result: ",result ,"truth: ",labels[i])
        if result != labels[i]:
            errorCount +=1
    print("error rate: ",errorCount/N)
    #   数据分析
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(dataSets[:10,0],dataSets[:10,1])     #两个特征
    plt.show()

#############################################     手写识别系统     #######################################################

def imgToVector(file_name):
    returnVector = np.zeros((1,1024))
    with open(file_name) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector
def testHandWriting(train_path, test_path):
    trainFileList = os.listdir(train_path)
    testFileList = os.listdir(test_path)
    N = len(trainFileList)
    trainMat = np.zeros((N,1024))
    trainLabels = []
    for index,name in enumerate(trainFileList):
        trainLabels.append(int(name.split("_")[0]))
        trainMat[index,:] = imgToVector(os.path.join(train_path,name))
    errorCount = 0
    for index, name in enumerate(testFileList):
        test_truth = int(name.split("_")[0])
        test_vector = imgToVector(os.path.join(test_path,name))
        result = knnClassifier(test_vector,trainMat,trainLabels,3)
        print("result: ",result ,"truth: ",test_truth)
        if result!=test_truth:
            errorCount +=1

    print("error rate: ",errorCount/len(testFileList))



if __name__ == "__main__":
    #testSimpleDatasets()
    #testDating()
    testHandWriting("/Users/androiduser/Desktop/Data/machinelearninginaction/Ch02/digits/trainingDigits",
                    "/Users/androiduser/Desktop/Data/machinelearninginaction/Ch02/digits/testDigits")
