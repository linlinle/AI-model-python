"""Decision Tree Regression with AdaBoos"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
X = np.linspace(1,6,100)[:,np.newaxis]
y = np.sin(X).ravel() + np.sin(6*X).ravel()+rng.normal(0,0.1,len(X))

regr1 = DecisionTreeRegressor(max_depth=5)
regr2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=300,random_state=rng)

regr1.fit(X,y)
regr2.fit(X, y)

# Predict
y_1 = regr1.predict(X)
y_2 = regr2.predict(X)

plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)

plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()