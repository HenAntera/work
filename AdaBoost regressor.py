# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:58:31 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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
y = dataset[:,[82]]

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

n_estim = 100
a = 1.5

clf = AdaBoostRegressor(n_estimators= n_estim, random_state=0, learning_rate= a, loss="square")
clf.fit(X_train, y_train.ravel())
scores = cross_val_score(clf, X, y.ravel(), cv=5)
print(np.mean(scores)) #-5.030471321130165
y_predict = clf.predict(X_test)
print(r2_score(y_test, y_predict)) #0.9876284403147888
print(mean_squared_error(y_test, y_predict)) #196889.58404675807
print(mean_squared_error(y_test, y_predict, squared = False)) #443.72241778701925

param_grid = {"n_estimators": [10, 15, 25, 50, 75, 100], "loss": ["linear", "square", "exponential"],
              "learning_rate": [0.01, 0.1, 1, 1.5, 2, 3]} 

grid = GridSearchCV(AdaBoostRegressor(), param_grid, cv=7)

grid.fit(X, y.ravel())

print(grid.best_params_)#{'learning_rate': 1.5, 'loss': 'square', 'n_estimators': 100}

#cm = confusion_matrix(y_test, y_predict)
#print(cm)


#c = np.arange(0.1, 5)
#train_score, val_score = validation_curve(SVC(), X, y.ravel(), "", c, cv=5)

#plt.plot(c, np.median(train_score, 1), color='blue', label='training score')
#plt.plot(c, np.median(val_score, 1), color='red', label='validation score')
#plt.legend(loc='best')
#plt.ylim(0, 1)
#plt.xlabel('C')
#plt.ylabel('score');

