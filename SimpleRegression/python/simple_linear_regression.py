# Simple Linear Regression
from __future__ import print_function, division
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data set
data_set = pd.read_csv('../Salary_Data.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values

# Split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get prediction
y_prediction = model.predict(X_train)

# Plot training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_prediction, color='blue')
plt.title('Salary vs Experience (Train set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()

# Plot test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y_prediction, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()


