# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def pca(dataMat, topK = 9999):
    meanVals = np.mean(dataMat,axis=0)

    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved,rowvar=0)               #协方差cov
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))    #计算矩阵特征向量
    eigValInd = np.argsort(eigVals)

    eigValInd = eigValInd[:-(topK +1):-1]
    redEigVects = eigVects[:,eigValInd]

    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat, reconMat

if __name__ == "__main__":
    dataMat = load_iris().data
    lowDMat, reconMat = pca(dataMat,topK=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten(), dataMat[:, 1].flatten(), marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()