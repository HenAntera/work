# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:17:29 2020

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from statistics import mean
from sklearn.metrics import f1_score
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

X = dataset[:,:79]
y = dataset[:,[83]]

X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

c=10

clf = SVC(C=c, degree=1, kernel= "sigmoid")
clf.fit(X_train, y_train.ravel())
scores = cross_val_score(clf, X, y.ravel(), cv=7)
print(mean(scores)) #0.5206098249576511
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm) # [[ 76  79]
          # [ 86 114]]
print(accuracy_score(y_test,y_pred)) #0.5352112676056338
print(recall_score(y_test, y_pred)) #0.57
print(f1_score(y_test, y_pred)) #0.5801526717557252

#c = np.arange(0.1, 20)
#train_score, val_score = validation_curve(SVC(), X, y.ravel(), "C", c, cv=5)

#plt.plot(c, np.median(train_score, 1), color='blue', label='training score')
#plt.plot(c, np.median(val_score, 1), color='red', label='validation score')
#plt.legend(loc='best')
#plt.ylim(0, 1)
#plt.xlabel('C')
#plt.ylabel('score');

#param_grid = {"C": [1, 5, 10, 15, 20,], "kernel": ["linear", "poly", "rbf", "sigmoid"],
              #"degree": [1, 3, 5, 7, 10]} 

#grid = GridSearchCV(SVC(), param_grid, cv=7)

#grid.fit(X, y.ravel())

#print(grid.best_params_)

#best_params_ = {'C': 10, 'degree': 1, 'kernel': 'sigmoid'}



