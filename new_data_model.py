# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:09:31 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import laplacian_kernel
from numpy import mean
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#%% Data
data = pd.read_csv("BTC.csv",header=0)

data["Unix Timestamp"] = pd.to_datetime(data["Unix Timestamp"],unit='s')

data = data.sort_values(by="Unix Timestamp")

data = data.drop(labels={"Date","Symbol", "Volume BTC"},axis=1)

data = data.drop(data.index[:5678])
data = data.drop(data.index[23590:])

statdf = data.describe()

dif = pd.read_csv("difficulty.csv",header=0)

dif['timestamp'] = pd.to_datetime(dif['timestamp'])

dif = dif.set_index('timestamp')

hrate = pd.read_csv("hash-rate.csv",header=0)

hrate['timestamp'] = pd.to_datetime(hrate['timestamp'])

hrate = hrate.set_index('timestamp')

market_cap = pd.read_csv("market-cap.csv",header=0)

market_cap['timestamp'] = pd.to_datetime(market_cap['timestamp'])

market_cap = market_cap.set_index('timestamp')

sopr = pd.read_csv("sopr.csv",header=0)

sopr['timestamp'] = pd.to_datetime(sopr['timestamp'])

sopr = sopr.set_index('timestamp')

fees_mean = pd.read_csv("fees-mean.csv",header=0)

fees_mean['timestamp'] = pd.to_datetime(fees_mean['timestamp'])

fees_mean = fees_mean.set_index('timestamp')

fees_total = pd.read_csv("fees-total.csv",header=0)

fees_total['timestamp'] = pd.to_datetime(fees_total['timestamp'])

fees_total = fees_total.set_index('timestamp')

act_addresses = pd.read_csv("active-addresses.csv",header=0)

act_addresses['timestamp'] = pd.to_datetime(act_addresses['timestamp'])

act_addresses = act_addresses.set_index('timestamp')

new_addresses = pd.read_csv("new-addresses.csv",header=0)

new_addresses['timestamp'] = pd.to_datetime(new_addresses['timestamp'])

new_addresses = new_addresses.set_index('timestamp')

tran_rate = pd.read_csv("transaction-rate.csv",header=0)

tran_rate['timestamp'] = pd.to_datetime(tran_rate['timestamp'])

tran_rate = tran_rate.set_index('timestamp')

tran_siz_mean = pd.read_csv("transaction-size-mean.csv",header=0)

tran_siz_mean['timestamp'] = pd.to_datetime(tran_siz_mean['timestamp'])

tran_siz_mean = tran_siz_mean.set_index('timestamp')

transfer_volume_mean = pd.read_csv("transfer-volume-mean.csv",header=0)

transfer_volume_mean['timestamp'] = pd.to_datetime(transfer_volume_mean['timestamp'])

transfer_volume_mean = transfer_volume_mean.set_index('timestamp')

market_cap_tether = pd.read_csv("market-cap-tether.csv",header=0)

market_cap_tether['timestamp'] = pd.to_datetime(market_cap_tether['timestamp'])

market_cap_tether = market_cap_tether.set_index('timestamp')

utxo_spent = pd.read_csv("utxo-value-spent-mean.csv",header=0)

utxo_spent['timestamp'] = pd.to_datetime(utxo_spent['timestamp'])

utxo_spent = utxo_spent.set_index('timestamp')

utxo_created = pd.read_csv("utxo-value-created-mean.csv",header=0)

utxo_created['timestamp'] = pd.to_datetime(utxo_created['timestamp'])

utxo_created = utxo_created.set_index('timestamp')

data = data.set_index('Unix Timestamp')
data = data.set_index(dif.index)

data["hrate"] = hrate["value"]
data["difficulty"] = dif["value"]
data["market_cap"] = market_cap["value"]
data["sopr"] = sopr["value"]
data["fees_mean"] = fees_mean["value"]
data["fees_total"] = fees_total["value"]
data["act_addresses"] = act_addresses["value"]
data["new_addresses"] = new_addresses["value"]
data["tran_rate"] = tran_rate["value"]
data["tran_siz_mean"] = tran_siz_mean["value"]
data["transfer_volume_mean"] = transfer_volume_mean["value"]
data["market_cap_tether"] = market_cap_tether["value"]
data["utxo_spent"] = utxo_spent["value"]
data["utxo_created"] = utxo_created["value"]
data["utxo_sum"] = utxo_created["value"] - utxo_spent["value"]

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume USD",fillna=True)

data = data.apply(pd.to_numeric, errors='coerce')

lst = list()

for i in data["others_dlr"]:
    if i <= 0 :
        lst.append(int(0))
    else:
        lst.append(int(1))

data["classification"] = lst
data["classification"] = data[["classification"]].shift(-1)

data["classification"] = data["classification"].fillna(0)

data = data.fillna(0)

#data = data.drop(labels = {'High','Low','volatility_kch','volatility_kcl','volatility_kcw','volatility_kcp','volatility_kchi',
 #                          'volatility_kcli','volatility_dcl', 'volatility_dch','volume_sma_em','volume_em','volume_vpt',
  #                         'trend_adx','trend_adx_pos','trend_adx_neg','trend_vortex_ind_pos','trend_vortex_ind_neg','trend_vortex_ind_diff',
   #                        'trend_trix','trend_cci','trend_dpo','trend_ichimoku_conv','trend_ichimoku_base','trend_ichimoku_a',
    #                       'trend_ichimoku_b','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_aroon_up','trend_aroon_down',
     #                      'trend_aroon_ind','momentum_uo','momentum_wr','momentum_roc'}, axis=1)
data = data.drop(labels = {'High','Low'}, axis=1)
dataset = data.values
dataset = dataset[42:,:]

X = dataset[:,:86]
y = dataset[:,[90]]
#%%feature selection
aaXX = np.delete(X, [6,7,20,33,34,35,35,41,42,44,55,56,57,58,76,77], axis=1)

X = aaXX[:,[2, 4, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 34, 
         37, 43, 45, 46, 47, 48, 49, 50, 58, 63, 64, 65, 66, 67, 68, 70]]
#%%
ope = aaXX[18838:,[0]]
clo = aaXX[18838:,[1]]
#%% Split
X_tra,X_te,y_train,y_test=train_test_split(X,y,test_size=0.2, shuffle=False, random_state=0)
#%% Processing
sc = StandardScaler()
#norm = Normalizer()

sel = VarianceThreshold(threshold=(0.8 * (1 - .8)))
X_trai = sel.fit_transform(X_tra)
X_tes =sel.transform(X_te)


X_train = sc.fit_transform(X_trai)
X_test = sc.transform(X_tes)

#X_train = norm.fit_transform(X_train)
#X_test = norm.transform(X_test)
#%% Model

c=120
gamma = 0.001
clf = SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced")
clf.fit(X_train, y_train.ravel())


y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
'''
print('Training set metrics:')
print('Accuracy:', accuracy_score(y_train, clf.predict(X_train)))
print('Precision:', precision_score(y_train, clf.predict(X_train)))
print('Recall:', recall_score(y_train, clf.predict(X_train)))
'''
print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))

print(cm)

print(f1_score(y_test, y_pred))

#%% Hyperparameter
splits = TimeSeriesSplit(n_splits=5)

#C_range = [40, 50, 60]
#gamma_range = [0.0001, 0.001, 0.01]

#param_grid = dict(gamma=gamma_range, C=C_range)

param_grid = {"C": [25, 50, 100, 120], "kernel": ["rbf", "sigmoid"], 
              "gamma": [0.0001, 0.001, 0.01], "class_weight":["balanced"],
              "cache_size":[200, 400, 600]} 

grid =RandomizedSearchCV(SVC(), param_grid, scoring = "precision", cv=splits, verbose=2, n_jobs=-1)

grid.fit(X_train, y_train.ravel())

print(grid.best_params_)

#%%Method of feature selection

sfs = SFS(SVC(), k_features=40, forward=True, 
          floating=False, scoring = 'accuracy', verbose=2, cv = splits, n_jobs=-1)

sfs = sfs.fit(X_train, y_train.ravel())

print(sfs.subsets_)
#%% Weights

# loop where the magic happens
# Agent
# relative agent profit
cum_sum = 1
# portfolio alocation
port_aloc = 0.05
win_factor = 1.1
lose_factor = 0.8
# win situation
def win(port_aloc):
    new_aloc = port_aloc * 1.5
    if new_aloc > 1:
        return 1
    else:
        return new_aloc
# lose situation
def lose(port_aloc):
    return port_aloc * 0.5

# recalculate alocation
def win_or_lose(gainz, pitaquepariu):
    if gainz > 0:
        return win(pitaquepariu)
    else:
        return lose(pitaquepariu)
    
# Agent decisions
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        order = 1
    else:
        order = -1    
    gains = ((clo[i]-ope[i])/ope[i])*order
    cum_sum = cum_sum*(1+gains*port_aloc)
    port_aloc = win_or_lose(gains, port_aloc)
    
print(cum_sum)
#%% Revenue
ret = list()
#cl = X_te[:,[1]]
#op = X_te[:,[0]]

for i in range(len(y_pred)):  
    gains = ((clo[i]-ope[i])/ope[i])*y_pred[i]
    ret.append(gains)
    
print(cum_sum)
print(np.mean(ret)*24*365)

#%% results from prediction

classify = np.vstack((y_train,y_pred[:, None]))
