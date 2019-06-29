# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:55:13 2019

@author: Dell
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer

#######################################

dataset = pd.read_csv('train.csv', sep = ',')
dataset = dataset.drop(['PassengerId', 'Name'], axis = 1)

dataset2 = pd.read_csv('test.csv', sep = ',')
ids = dataset2.iloc[:, 0].values
ids = np.reshape(ids,(len(ids),1))
dataset2 = dataset2.drop(['PassengerId', 'Name'], axis = 1)

X_train = pd.DataFrame(dataset.iloc[:, 1:].values)
y_train = pd.DataFrame(dataset.iloc[:, 0].values)
X_test = dataset2



#######################################
string_cols = []
integer_cols = []
decimal_cols = []

for i in range(X_train.shape[1]):
    dtype = type(X_train[i][0])
    if dtype == str:
        string_cols.append(i)
    elif dtype == int:
        integer_cols.append(i)
    else:
        decimal_cols.append(i)
#Classified column 7 as decimal but it is string, so hard coded to be string column:
string_cols.append(decimal_cols[-1])
decimal_cols = decimal_cols[0:-1]
num_cols = list(np.concatenate((integer_cols, decimal_cols)))
#######################################
#fill NAN floats with mean
#fill NAN integers with median
#fill NAN strings with most frequent

imputer_X = SimpleImputer(missing_values = np.nan, strategy = "median")
X_train = X_train.values
X_test = X_test.values
X_train[:, integer_cols] = imputer_X.fit_transform(X_train[:, integer_cols])
X_test[:, integer_cols] = imputer_X.fit_transform(X_test[:, integer_cols])

imputer_X = SimpleImputer(missing_values = np.nan, strategy = "mean")
X_train[:, decimal_cols] = imputer_X.fit_transform(X_train[:, decimal_cols])
X_test[:, decimal_cols] = imputer_X.fit_transform(X_test[:, decimal_cols])

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

most_frequent_train = X_train[string_cols].mode()
most_frequent_test = X_test[string_cols].mode()

for i in string_cols:
    X_train[i] = X_train[i].fillna(value = most_frequent_train[i][0])
    X_test[i] = X_test[i].fillna(value = most_frequent_test[i][0])
    
#######################################################
#Scaling and encoding
temp = []
for i in string_cols:
    temp.append(X_train[i].unique())
for i in string_cols:
    temp.append(X_test[i].unique())


temp2 = []
for i in range(len(temp)):
    for j in range(len(temp[i])):
        temp2.append(temp[i][j])
temp2 = set(temp2)
temp2 = list(temp2)


labelencoder_X = LabelEncoder()
labelencoder_X.fit(temp2)


for i in string_cols:
    X_train[i] = labelencoder_X.transform(X_train[i])
    X_test[i] = labelencoder_X.transform(X_test[i])

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

###################################################
#Random Forests
#Model2
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
#prediction

predicted_targets = classifier.predict(X_test)
predicted_targets = np.reshape(predicted_targets,(len(predicted_targets),1))

predicted_targets = np.concatenate((ids, predicted_targets), axis = 1)
submission = pd.DataFrame({'PassengerId':predicted_targets[:,0],'Survived':predicted_targets[:,1]})
submission.to_csv('Predictions_RF.csv')