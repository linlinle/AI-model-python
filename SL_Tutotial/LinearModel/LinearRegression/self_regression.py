
import numpy as np
from sklearn.datasets import load_boston,load_diabetes
from sklearn.metrics import mean_squared_error




def load_DataSet():

    diabetes = load_diabetes()
    diabetes_X = diabetes.data
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    return diabetes_X_train,diabetes_X_test,diabetes_y_train[:,np.newaxis],diabetes_y_test[:,np.newaxis]

def standRegression(X, y):
    XMat = np.mat(X)
    yMat = np.mat(y)
    xTx = XMat.T*XMat
    if np.linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    weights = xTx.I * (XMat.T*yMat)
    return weights

if __name__ == "__main__":
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = load_DataSet()
    weights = standRegression(diabetes_X_train,diabetes_y_train)
    print(weights)
    pred = np.dot(diabetes_X_test, weights)
    print(mean_squared_error(diabetes_y_test,pred))