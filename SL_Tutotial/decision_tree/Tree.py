
from math import log
from collections import Counter


def creatSimpleDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def informationEntropy(dataset):
    """
    计算数据集的信息熵
    :param dataset:
    :return:
    """
    N = len(dataset)    #   数据集实例总数
    labelsCount = dict(Counter(list(zip(*dataset))[-1]))
    entroy = 0.0
    for key in labelsCount:
        prob = float(labelsCount[key])/N
        entroy -= prob*log(prob,2)
    return entroy

def splitDataSet(dataSet, a, v):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分数据集D
    :param axis: 划分数据集的特征a
    :param value: 特征取值v
    :return: D中特征a==v的数据子集，并删除特征a
    """
    resDataset = []
    for example in dataSet:
        if example[a] == v:
            resexample = example[:a]
            resexample.extend(example[a+1:])
            resDataset.append(resexample)
    return resDataset

def chooseBestFeature(dataset):
    """
    选择最优划分特征

    :param dataSet: 二维列表，N*M(examples*features)
    :return: 最优特征的索引
    """
    N = len(dataset)
    M = len(dataset[0])-1
    baseEntory = informationEntropy(dataset)
    baseInfGain = 0.0;bestFeature = -1
                                                                                             # 遍历所有特征
    for i in range(M):
        oneFeatureList = [example[i] for example in dataset]    #选择一个特征
        uniqueVals = set(oneFeatureList)                        #特征可能取值
        newEntropy = 0.0
        for value in uniqueVals:                                                           # 遍历所有可能取值
            subDataSet = splitDataSet(dataset,i , value)
            prob = len(subDataSet)/N
            newEntropy += prob * informationEntropy(subDataSet)
        infoGain =baseEntory - newEntropy
        if infoGain > baseInfGain:
            baseInfGain = infoGain
            bestFeature = i
    return bestFeature

def majorityVote(classList):
    """
    多数表决
    :param classList:
    :return:
    """
    classCount = dict(Counter(classList))
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def creatTree(dataset,lables):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):         #类别相同，停止划分
        return classList[0]
    if len(dataset[0]) ==1:                                     #特征集为1，返回出现最多的类别
        return majorityVote(classList)
    bestFeature = chooseBestFeature(dataset)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del labels[bestFeature]
    featureValues = [example[bestFeature] for example in dataset]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = creatTree(splitDataSet(dataset,bestFeature,value),subLabels)
    return myTree


if __name__ == "__main__":
    simpleDataset, labels = creatSimpleDataset()
    myTree = creatTree(simpleDataset,labels)
    print(myTree)