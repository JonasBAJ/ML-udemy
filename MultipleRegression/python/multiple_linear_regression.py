# Multiple Linear Regression
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import pandas as pd

# Importing the data set
data_set = pd.read_csv('50_Startups.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values

# Encoding categorical data
X[:, 3] = LabelEncoder().fit_transform((X[:, 3]))
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap (bc constant term covers one of the dummies)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
model = LinearRegression().fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Backward Elimination
X = np.append(np.ones((50,1)).astype(int), X, 1)
X_opt = X[:, [0 ,1, 2, 3, 4, 5]]
model_OLS = sm.OLS(y, X_opt).fit()
model_OLS.summary()

X_opt = X[:, [0 ,1, 3, 4, 5]]
model_OLS = sm.OLS(y, X_opt).fit()
model_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
model_OLS = sm.OLS(y, X_opt).fit()
model_OLS.summary()

X_opt = X[:, [0, 3, 5]]
model_OLS = sm.OLS(y, X_opt).fit()
model_OLS.summary()

X_opt = X[:, [0, 3]]
model_OLS = sm.OLS(y, X_opt).fit()
model_OLS.summary()