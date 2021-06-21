# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:15:28 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

data = pd.read_csv("database.csv", sep=";",header=0, index_col=0)   

statistics = data.describe()

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume",fillna=True)

lst = list()

forecast = 1
data["prediction"] = data[["Close"]].shift(-forecast)
data.drop(data.tail(forecast).index, inplace=True)

for i in data["others_dlr"]:
    if i <= 0 :
        lst.append(int(0))
    else:
        lst.append(int(1))

data["classification"] = lst
data["classification"] = data[["classification"]].shift(-1)

data["classification"] = data["classification"].fillna(0)

dataset = data.values

dataset = dataset[42:,:]

X = dataset[:,:9]
y = dataset[:,[83]]

X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

c=3

clf = SVC(kernel="sigmoid", C=c)
clf.fit(X_train, y_train.ravel())
scores = cross_val_score(clf, X, y.ravel(), cv=5)
print(scores)
y_predict = clf.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)