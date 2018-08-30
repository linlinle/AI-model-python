import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tools.eval_measures import rmse

def result_loss(loss = 'lad'):
    mod = GradientBoostingRegressor(loss=loss)
    fiter = mod.fit(train_x, train_y)
    predict = fiter.predict(test_x)

    print("Loss -> %f" % rmse(predict, test_y))

def add_outlier(train_x,train_y):
    stats = data.describe()
    extremes = stats.loc[['min', 'max'], :].drop('medv', axis=1)

    # Generate 5 outliers
    np.random.seed(1234)
    rands = np.random.rand(5, 1)
    min_array = np.array(extremes.loc[['min'], :])
    max_array = np.array(extremes.loc[['max'], :])
    rang = max_array - min_array
    outliers_x = (rands * rang) + min_array
    # Change the type of 'chas', 'rad' and 'tax' to rounded of Integers
    outliers_x[:, [3, 8, 9]] = np.int64(np.round(outliers_x[:, [3, 8, 9]]))
    medv_outliers = np.array([0, 0, 600, 700, 600])

    train_x = np.append(train_x, outliers_x, axis=0)
    train_y = np.append(train_y, medv_outliers, axis=0)
    return train_x,train_y

if __name__ == "__main__":

    # Make pylab inline and set the theme to 'ggplot'
    plt.style.use('ggplot')

    data = pd.read_csv('Housing.csv')

    data_indep = data.drop('medv', axis=1)
    data_dep = data['medv']

    train_x, test_x,train_y, test_y = train_test_split(data_indep,data_dep,
                                                       test_size=0.20,
                                                       random_state=42)

    ############################Regression without any Outliers:##########################################
    result_loss()

    result_loss('ls')

    # L1 = 3.595860 > L2=2.492961

    ##############################Regression with Outliers##############################################

    train_x, train_y = add_outlier(train_x,train_y)
    fig = plt.figure(figsize=(13,7))
    plt.hist(train_y, bins=50, range=(-10,800))
    fig.suptitle('medv Count', fontsize=16)
    plt.xlabel('medv', fontsize=16)
    plt.ylabel('count',fontsize=16)
    plt.show()



    result_loss()
    result_loss('ls')
    # L1 = 4.333808 > L2=13.294986