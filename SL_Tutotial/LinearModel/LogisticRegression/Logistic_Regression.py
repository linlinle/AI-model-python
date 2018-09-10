
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset():
    iris = load_iris()
    dataMat ,labelList= iris.data[:100,:2],iris.target[:100,np.newaxis]
    X_train = np.vstack((dataMat[:40],dataMat[60:]))
    X_test = dataMat[40:60]
    y_train = np.vstack((labelList[:40],labelList[60:]))
    y_test = labelList[40:60]
    return X_train,X_test,y_train,y_test


class self_LogisticRegression():
    def __init__(self):
        self.num_iter = 500

    def _sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))


    def _gradient_ascent(self,dataMat, classList):
        """
        梯度上升与梯度下降类似，一个求最大值，一个求最小值
        :param dataMat:
        :param classList:
        :return:
        """
        example_num,feature_num = np.shape(dataMat)

        dataMarix_b = np.hstack((np.ones((example_num,1)),dataMat))
        weights = np.zeros((feature_num+1,1))
        weightsTemp = np.zeros((feature_num+1,1))
        for i in range(self.num_iter):
            alpha = 4 / (1.0 + i) + 0.01
            predict = self._sigmoid(np.dot(dataMarix_b,weights))            #np.mat不需要求和
            Matrierror =  predict-classList
            for j in range(feature_num+1):
                matrixSumTerm = np.reshape(dataMarix_b[:, j],(1,example_num))
                weightsTemp[j] = weights[j] - alpha / example_num * np.dot(matrixSumTerm,Matrierror)
            weights = weightsTemp
        return weights

    def _stochastic_gradient_ascent(self,dataMat, classList):
        """

        :param dataMat:
        :param classList:
        :param numIter: 迭代次数
        :return:
        """
        example_num, feature_num = np.shape(dataMat)

        dataMarix_b = np.hstack((np.ones((example_num, 1)), dataMat))
        weights = np.zeros((feature_num + 1, 1))
        weightsTemp = np.zeros((feature_num + 1, 1))
        for j in range(self.num_iter):
            dataInex = list(range(example_num))
            for i in range(example_num):
                alpha = 4/(1.0+j+i)+0.01                                # alpha每次迭代需要调整
                randIndex = int(np.random.uniform(0,len(dataInex)))     #随机选取更新
                predict = self._sigmoid(np.dot(dataMarix_b[randIndex],weights))    #np,array需要求和
                error = predict- classList[randIndex,:]
                for j in range(feature_num + 1):
                    weightsTemp[j] = weights[j] - alpha * error*dataMarix_b[randIndex][j]
                weights = weightsTemp
                del (dataInex[randIndex])
        return weights

    def fit(self,X_train, Y):
        weights = self._stochastic_gradient_ascent(X_train,Y)
        self.coef_ = weights

    def predict(self,X_test):
        pred = np.dot(X_test,self.coef_[1:]) + self.coef_[0]
        return pred

def plotBestFit(weights):
    X_train, X_test, y_train, y_test = load_dataset()
    dataArr = np.vstack((X_train,X_test))
    labelMat = np.vstack((y_train,y_test))
    n = np.shape(dataArr)[0]
    xcord1,ycord1,xcord2,ycord2 = [],[],[],[]
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(4,8,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y.transpose())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ =="__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    self_model = self_LogisticRegression()
    self_model.fit(X_train,y_train)
    y_pred = self_model.predict(X_test)
    print('Coefficients: \n', self_model.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    plotBestFit(self_model.coef_)


    plt.show()
    SL_model = LogisticRegression()
    SL_model.fit(X_train,y_train)
    y_pred = SL_model.predict(X_test)
    print('Coefficients: \n', SL_model.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    SL_weights = [SL_model.intercept_[0],SL_model.coef_[0,0],SL_model.coef_[0,1]]
    plotBestFit(SL_weights)


