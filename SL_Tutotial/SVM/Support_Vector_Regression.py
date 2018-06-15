# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5*np.random.randn(40,1),axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] +=3*(0.5 - np.random.rand(8)) #[::5]=[0:end:5]从0开始每间隔5个数

# #############################################################################
# Fit regression Re_classifying
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_lin = SVR(kernel='linear',C = 1e3)
svr_poly = SVR(kernel='poly',C = 1e3, degree=2)
svr_rbf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)

# #############################################################################
# Look at the results
lw =2
plt.scatter(X,y,color='darkorange', label='data')
plt.plot(X, y_rbf,color='navy', lw=lw, label='RBF Re_classifying')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear Re_classifying')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial Re_classifying')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()