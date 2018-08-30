"""判断侮辱恶意词汇"""

import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is 侮辱性文字, 0 is 正常文字
    return postingList,classVec

def creatVocabList(dataset):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataset:
    :return:
    """
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setWord2vec(vocabList, inputset):
    """
    根据vocabList的索引位置，表示单词是否出现，1出现，0未出现
    :param vocabList: 词汇表
    :param inputset: 输入文本
    :return:
    """
    returnvec = [0]*len(vocabList)
    for word in inputset:
        if word in vocabList:
            returnvec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnvec

def bagOfWords2VecMN(vocabList, inputSet):
    """
    由于一个词在文档中出现次数不止一次，所以此函数每遇见一个单词，就会增加词向量中的对应值，而不是一直设为1
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB_0(trainMatrix, trainCategory):
    """
    根据训练样本，求出先验类概率密度函数，每个单词为侮辱性词语的概率p1Vect，每个单词不是侮辱词语的概率p0Vect，文档属于侮辱类的概率pAbusive
    :param trainMatrix: 文档矩阵
    :param trainCategory: 文档矩阵对应的标签向量
    :return: p(x|0),p(x|1),p(1)
    """
    numTrainLines, numWords = len(trainMatrix),len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainLines)

    # 计算联合概率密度时，乘积元素不能为0，否则结果恒为0，将所有词出现次数初始化为1，分母初始化为2
    p0Num,p1Num = np.ones(numWords), np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0

    # 统计训练样本中每个单词在侮辱和非侮辱标签中出现的次数
    for i in range(numTrainLines):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 防止下溢
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify: 待分类的词汇索引向量
    :param p0Vec: p(x|1)概率密度
    :param p1Vec:  p(x|0)概率密度
    :param pClass1:  p(1)
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    listPosts, listLabel = loadDataSet()
    myVocabList = creatVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setWord2vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB_0(trainMat,listLabel)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setWord2vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setWord2vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
