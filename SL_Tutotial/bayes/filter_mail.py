"""过滤垃圾邮件"""
import re
import numpy as np
from SL_Tutotial.bayes.naive_bayes import classifyNB,trainNB_0,creatVocabList,bagOfWords2VecMN

def textParse(bigstring):
    #  正则表达式切分句子
    listTokens = re.split(r'\W*',bigstring)
    # 字符长度大于2，全部小写
    return [tok.lower() for tok in listTokens if len(tok) > 2]

if __name__ == "__main__":
    docList, classList,fullText = [],[],[]
    # 数据集
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = creatVocabList(docList)  # create vocabulary
    trainingSet = np.arange(50)
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        trainingSet = np.delete(trainingSet, randIndex)
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB_0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))