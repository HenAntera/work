# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:25:20 2020

@author: Henrique Oliveira
"""
%matplotlib inline
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

k=4
n_estim = 100
a = 0.001
c= 0.5

names = ["Nearest Neighbors", "Sigmoid SVM", "RBF SVM", "AdaBoost"]

classifiers = [
    KNeighborsClassifier(k),
    SVC(kernel="sigmoid", C=c),
    SVC(C=c),
    AdaBoostClassifier(n_estimators= n_estim, random_state=0, learning_rate= a)]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train.ravel())
    scores = cross_val_score(clf, X, y.ravel(), cv=5)
    print(scores)


X = dataset[:,:79]
y = dataset[:,[83]]

split_percent = 0.80
split = int(split_percent*len(y))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

lol = data[{'Open','Volume','market_cap','total_volume','Value - Hrate','Value - Uadresses',
            'volume_fi','momentum_mfi','volume_sma_em','volatility_bbm','volatility_kcp','volatility_kcli',
            'volatility_dch','trend_macd_diff','trend_sma_fast','trend_ema_fast','trend_adx','trend_adx_pos',
            'trend_adx_neg','trend_vortex_ind_neg','trend_mass_index','trend_ichimoku_base',
            'trend_psar_up_indicator','momentum_stoch'}]

def retur_pro(sign, cl, hours):
    S = list()
    #len(sign)
    for i in range(0, len(sign), hours):
        end_i = i + hours
        enter = 0
        leave = 0
        # check if we are beyond the dataset
        if end_i > len(sign):
            #print("A")
            break
        c = sign[i]
        #print("c:" + str(c))
        #print("i:" + str(i) + " end_i:" + str(end_i))
        #print("INI:" + str(cl[i]) + " FIN:" + str(cl[end_i]))
          
        if c == 1:
            S.append(0)
            
        if c == 0:
            flag_condition = False
            for e in cl[i:end_i]:
                if e >= cl[i]*1.05:
                    S.append(-0.05)
                    flag_condition=True
                    break
                if e <= cl[i]*0.95:
                    S.append(0.05)
                    flag_condition=True
                    break
            if not flag_condition:   
                enter = cl[i]
                leave = cl[end_i]
                r = -(leave-enter)/enter
                S.append(r)
                                        
        if c == 2:
            flag_condition = False
            for e in cl[i:end_i]:
                if e >= cl[i]*1.05:
                    S.append(0.05)
                    flag_condition=True
                    break
                if e <= cl[i]*0.95:
                    S.append(-0.05)
                    flag_condition=True
                    break
            if not flag_condition:   
                enter = cl[i]
                leave = cl[end_i]
                r = (leave-enter)/enter
                S.append(r)

    return array(S)





