# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:56:55 2017

@author: Sherin
"""
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
label=[[10],[20],[30],[40],[50],[60],[70],[80],[90],[100]]
data=[[1,1,1,1,1,1,1,1,1,1],[1,2,2,2,3,3,4,4,4,5],[5,6,10,12,17,12,6,5,7,10]]

# Split the data into training/testing sets
diabetes_X_train = np.transpose(data)
diabetes_X_test = np.transpose(data)

# Split the targets into training/testing sets
diabetes_y_train = label
diabetes_y_test = label

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

clf = Ridge(alpha=0.1)
clf.fit(diabetes_X_train, diabetes_y_train) 
 # Make predictions using the testing set
diabetes_y_pred = clf.predict(diabetes_X_test)
# The coefficients
print('Coefficients: \n', clf.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))