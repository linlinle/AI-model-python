
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_DataSet():

    diabetes = load_diabetes()
    diabetes_X = diabetes.data[:,np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    return diabetes_X_train,diabetes_X_test,diabetes_y_train,diabetes_y_test

def feature_scaling(dataSet):
    data_arr = np.array(dataSet)

    mu = np.mean(data_arr,0)
    sigma = np.std(data_arr,0)
    for i in range(data_arr.shape[1]):
        data_arr[:,i] = (data_arr[:,i]-mu[i])/sigma[i]
    dataMatrix = np.hstack((np.ones((dataSet.shape[0],1)),data_arr)) # # 在X前加一列1,参数b
    return dataMatrix

class self_LinearRegression():
    def __init__(self,alpha=1, num_iter = 400):
        self.alpha = alpha
        self.num_iter = num_iter



    def _gradient_descent(self,dataSet, classlist):
        example_num, feature_num = dataSet.shape
        # 添加参数b及其数据集
        dataSet_b = np.hstack((np.ones((example_num,1)),dataSet))
        weights = np.zeros((feature_num+1,1))
        weightsTemp = np.matrix(np.zeros((feature_num+1,1)))

        for i in range(self.num_iter):
            matrixError = np.dot(dataSet_b,weights)-classlist[:,np.newaxis]
            for j in range(feature_num+1):
                matrixSumTerm = np.reshape(dataSet_b[:, j],(1,422))
                weightsTemp[j] = weights[j] - self.alpha / example_num * np.dot(matrixSumTerm,matrixError)
            weights = weightsTemp

        return weights

    def fit(self,X_train, Y):
        weights = self._gradient_descent(X_train,Y)
        self.coef_ = weights

    def predict(self,X_test):
        pred = np.dot(X_test,self.coef_[1:]) + self.coef_[0]
        return pred

if __name__ =="__main__":

    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = load_DataSet()

    SL_model = LinearRegression()
    SL_model.fit(diabetes_X_train,diabetes_y_train)
    diabetes_y_pred = SL_model.predict(diabetes_X_test)
    print('Coefficients: \n', SL_model.coef_,SL_model.intercept_)
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)


    plt.show()

    self_model = self_LinearRegression()
    self_model.fit(diabetes_X_train,diabetes_y_train)
    diabetes_y_pred =self_model.predict(diabetes_X_test)
    print('Coefficients: \n', self_model.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    plt.scatter(diabetes_X_test, diabetes_y_test, color='red')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)


    plt.show()

