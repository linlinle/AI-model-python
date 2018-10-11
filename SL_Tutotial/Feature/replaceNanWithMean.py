# -*- coding: utf-8 -*-
import numpy as np
def replaceNanWithMean(dataMat):
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(np.isnan(dataMat[:,i]))[0],i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i]))[0], i] = meanVal
    return dataMat