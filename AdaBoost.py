# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:22:12 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit

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

lol = data[{'Open','Volume','market_cap','total_volume','Value - Hrate','Value - Uadresses',
            'volume_fi','momentum_mfi','volume_sma_em','volatility_bbm','volatility_kcp','volatility_kcli',
            'volatility_dch','trend_macd_diff','trend_sma_fast','trend_ema_fast','trend_adx','trend_adx_pos',
            'trend_adx_neg','trend_vortex_ind_neg','trend_mass_index','trend_ichimoku_base',
            'trend_psar_up_indicator','momentum_stoch','volatility_bbhi','volatility_bbli','volatility_kchi',
            'trend_psar_down_indicator'}]

dataset = data.values
lolset = lol.values
lolset =lolset[42:,:]
X = lolset
dataset = dataset[42:,:]

X = dataset[:,:79]
y = dataset[:,[83]]

X = X[:,[0, 8, 9, 21, 28, 50, 74]]
#X = X[:,[0, 3, 6, 8, 10, 13, 14, 24, 30, 32, 36, 38, 39, 40, 44, 45, 46, 48, 49, 50, 
 #        51, 52, 53, 54, 56, 61, 65, 66, 71, 72, 73, 78]]
#poly = PolynomialFeatures(interaction_only=True)
#X = poly.fit_transform(X)

#X = X[:,[15, 22, 36, 40, 41, 43, 49, 107, 110, 115, 119, 122, 123, 127, 134, 142, 147, 184, 190, 203, 207, 210, 211, 216, 218,
 #        220, 221, 222, 224, 227, 230, 232, 255, 258, 264, 288, 295, 299, 302, 303, 309, 318, 334, 357, 379, 387, 388, 390]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, shuffle=False, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_estim = 40
a = 0.15

clf = AdaBoostClassifier(base_estimator=RidgeClassifierCV(), algorithm="SAMME", n_estimators=n_estim, learning_rate=a)
clf.fit(X_train, y_train.ravel())
#scores = cross_val_score(clf, X, y.ravel(), cv=10, scoring="accuracy")
#print(np.mean(scores))
y_pred = clf.predict(X_test)
'''
df = pd.DataFrame (y_pred)

filepath = 'file.xlsx'

df.to_excel(filepath, index=False)
'''
cm = confusion_matrix(y_test, y_pred)


print('Training set metrics:')
print('Accuracy:', accuracy_score(y_train, clf.predict(X_train)))
print('Precision:', precision_score(y_train, clf.predict(X_train)))
print('Recall:', recall_score(y_train, clf.predict(X_train)))

print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))

print("Confusion Matrix")
print(cm)
print("f1 Score",f1_score(y_test, y_pred))

splits = TimeSeriesSplit(n_splits=5)
'''
param_grid = {"n_estimators": [40, 50, 60, 70], 
              "base_estimator":[GradientBoostingClassifier(), RidgeClassifierCV(), DecisionTreeClassifier(), ExtraTreesClassifier()],
              "learning_rate": [0.05, 0.1, 0.15]} #{'learning_rate': 2, 'loss': 'square', 'n_estimators': 75}

grid = GridSearchCV(AdaBoostClassifier(algorithm='SAMME'), param_grid, cv=splits, verbose=2)

grid.fit(X_train, y_train.ravel())

print(grid.best_params_)
'''
#clf = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(), algorithm="SAMME", n_estimators=n_estim, random_state=0, learning_rate=a)
#clf.fit(X_train, y_train.ravel())
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=splits)
print("Cross-Validation",np.mean(scores))


