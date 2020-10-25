# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:25:20 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

data = pd.read_csv("database.csv", sep=";",header=0,index_col=0)   

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume",fillna=True)

lst = list()

for i in data["others_dlr"]:
    if i <= 0 :
        lst.append(int(0))
    else:
        lst.append(int(1))

data["classification"] = lst
forecast = 1
data["prediction"] = data[["Close"]].shift(-forecast)
data.drop(data.tail(forecast).index, inplace=True)

dataset = data.values

X = dataset[42:,:10]
y = dataset[42:,[3]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

regr = AdaBoostRegressor()

regr.fit(X_train,y_train.ravel())

print(regr.score(X_test,y_test))

y_predict = regr.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_predict, y_test)
print(mse)