# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:39:52 2020

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
import seaborn as sns
import statsmodels.api as sm

data = pd.read_csv("database.csv", sep=";",header=0, index_col=0)   

statistics = data.describe()

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume",fillna=True)

data = data.apply(pd.to_numeric, errors='coerce')

features = data.drop(labels={"Close",'trend_psar_down','total_volume',
                              'trend_aroon_up','momentum_stoch','trend_aroon_ind',
                              'momentum_rsi','trend_adx','market_cap',"others_dr","others_dlr","others_cr"}, axis=1)

features['Close'] = data['Close']

X_reg = features.drop(labels={'Close'}, axis=1)
Y_reg = features['Close']

X_train,X_test,y_train,y_test=train_test_split(X_reg,Y_reg,test_size=0.2)

X=StandardScaler().fit_transform(X_train)


model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#colormap = plt.cm.inferno
#plt.figure(figsize=(100,50))
#corr = features.corr()
#sns.heatmap(corr[corr.index == 'Close'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
#plt.show()

#sns.set_theme(style="white")


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

#X = dataset[:,:9]
#y = dataset[:,[83]]

#X=StandardScaler().fit_transform(X)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#k=4

