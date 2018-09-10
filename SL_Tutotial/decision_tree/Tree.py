
from math import log
from collections import Counter


def creatSimpleDataset():

    dataSet_label = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feature_name = ['no surfacing', 'flippers']
    return dataSet_label, feature_name

def informationEntropy(dataSet_label):
    """
    计算数据集的信息熵
    :param dataset: 输入数据集，最后一列包含类别信息
    :return: 此数据的信息熵
    """
    N = len(dataSet_label)    #   数据集实例总数
    classCount = dict(Counter(list(zip(*dataSet_label))[-1]))
    entroy = 0.0
    for key in classCount:
        prob = float(classCount[key])/N
        entroy -= prob*log(prob,2)
    return entroy

def splitDataSet(dataSet_label, a, v):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分数据集D
    :param axis: 划分数据集的特征a
    :param value: 特征取值v
    :return: D中特征a==v的数据子集，并删除特征a
    """
    resDataset = []
    for example in dataSet_label:
        if example[a] == v:
            resexample = example[:a]
            resexample.extend(example[a+1:])
            resDataset.append(resexample)
    return resDataset

def chooseBestFeature(dataSet_label):
    """
    选择最优划分特征

    :param dataSet: 二维列表，N*(M+1) (examples*features，example*1)
    :return: 最优特征的索引
    """
    N = len(dataSet_label)
    M = len(dataSet_label[0])-1
    baseEntory = informationEntropy(dataSet_label)
    baseInfGain = 0.0;bestFeature = -1
                                                                                             # 遍历所有特征
    for i in range(M):
        oneFeatureList = [example[i] for example in dataSet_label]    #选择一个特征
        uniqueVals = set(oneFeatureList)                        #特征可能取值
        newEntropy = 0.0
        for value in uniqueVals:                                                           # 遍历所有可能取值
            subDataSet = splitDataSet(dataSet_label,i , value)
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
    :param classList: 节点的所有样本类别信息
    :return: 样本次数出现最多的类别
    """
    classCount = dict(Counter(classList))
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def creatTree(dataSet_label, feature_name):
    """

    :param dataset: 数据集【特征+类别】
    :param lables: 特征名称列表
    :return: 字典形式的树结构
    """
    classList = [example[-1] for example in dataSet_label]
    if classList.count(classList[0]) == len(classList):         #类别相同，停止划分
        return classList[0]
    if len(dataSet_label[0]) ==1:                                     #特征集为1，返回出现最多的类别
        return majorityVote(classList)
    bestFeature = chooseBestFeature(dataSet_label)
    bestFeatureLabel = feature_name[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del feature_name[bestFeature]
    featureValues = [example[bestFeature] for example in dataSet_label] # 列表包含所有属性集
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = feature_name[:]
        myTree[bestFeatureLabel][value] = creatTree(splitDataSet(dataSet_label,bestFeature,value),subLabels)
    return myTree


if __name__ == "__main__":
    dataSet_label, feature_name = creatSimpleDataset()
    myTree = creatTree(dataSet_label, feature_name)
    print(myTree)