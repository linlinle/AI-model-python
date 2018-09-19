"""该示例说明并比较single estimator的预期均方误差与bagging ensemble的偏差 - 方差分解。"""
import numpy as np
import matplotlib.pylab as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

def f(x):
    x = x.ravel()
    return np.exp(-x**2)+1.5*np.exp(-(x -2)**2)

def gendrate(n_sample,noise,n_repeat=1):
    X = np.random.rand(n_sample)*10-5
    X=np.sort(X)
    if n_repeat ==1:
        y = f(X)+np.random.normal(0.0,noise,n_sample)
    else:
        y = np.zeros((n_sample, n_repeat))

        for i in range(n_repeat):
            y[:,i] = f(X)+np.random.normal(0.0,noise,n_sample)

    X = X.reshape((n_sample,1))
    return X,y

if __name__ == '__main__':

    n_repeat = 50  # 迭代次数
    n_train = 50  # 训练集大小
    n_test = 1000  # 测试集大小
    noise = 0.1
    np.random.seed(0)

    estimator = [("Tree", DecisionTreeRegressor()),
                 ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]
    n_estimator = len(estimator)

    X_train, y_train = [],[]
    for i in range(n_repeat):
        X,y = gendrate(n_sample=n_train, noise=noise)
        X_train.append(X)
        y_train.append(y)

    X_test, y_test = gendrate(n_sample=n_test,noise=noise,n_repeat=n_repeat)

    for n,(name, estimator) in enumerate(estimator):
        y_predict = np.zeros((n_test,n_repeat))

        for i in range(n_repeat):
            estimator.fit(X_train[i],y_train[i])
            y_predict[:,i] = estimator.predict(X_test)
        y_error = np.zeros(n_test)
        for i in range(n_repeat):
            for j in range(n_repeat):
                y_error += (y_test[:, j] - y_predict[:, i]) ** 2
        y_error /= (n_repeat*n_repeat)

        y_noise = np.var(y_test, axis=1)
        y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
        y_var = np.var(y_predict, axis=1)

        print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
              " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                          np.mean(y_error),
                                                          np.mean(y_bias),
                                                          np.mean(y_var),
                                                          np.mean(y_noise)))

        # Plot figures
        plt.subplot(2, n_estimator, n + 1)
        plt.plot(X_test, f(X_test), "b", label="$f(x)$")
        plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

        for i in range(n_repeat):
            if i == 0:
                plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
            else:
                plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

        plt.plot(X_test, np.mean(y_predict, axis=1), "c",
                 label="$\mathbb{E}_{LS} \^y(x)$")

        plt.xlim([-5, 5])
        plt.title(name)

        if n == 0:
            plt.legend(loc="upper left", prop={"size": 11})

        plt.subplot(2, n_estimator, n_estimator + n + 1)
        plt.plot(X_test, y_error, "r", label="$error(x)$")
        plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
        plt.plot(X_test, y_var, "g", label="$variance(x)$"),
        plt.plot(X_test, y_noise, "c", label="$noise(x)$")

        plt.xlim([-5, 5])
        plt.ylim([0, 0.1])

        if n == 0:
            plt.legend(loc="upper left", prop={"size": 11})

plt.show()