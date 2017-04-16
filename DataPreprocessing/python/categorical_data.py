# Data Pre-Processing

from __future__ import print_function

# Importing the libraries
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class Helpers(object):

    @staticmethod
    def print_matrix(matrix, f_round=0):
        for r in matrix:
            for c in r:
                if isinstance(c, float): print (round(c, f_round), end='')
                else: print(c, end='')
            print()


# Importing the data set
data_set = pd.read_csv('../Data.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values


# Taking care of missing data
X[:, 1:3] = Imputer(missing_values='NaN', strategy='median', axis=0)\
    .fit(X[:, 1:3])\
    .transform(X[:, 1:3])


# Encoding the categorical Independent Variable
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])

X = OneHotEncoder(categorical_features=[0])\
    .fit_transform(X)\
    .toarray()

# Encoding the Dependent Variable
y = LabelEncoder().fit_transform(y)

# Separate variables into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize variables
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit(X_test)


print(X_train)
print(X_test)

print(y_train)
print(y_test)
