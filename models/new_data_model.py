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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from sklearn.metrics import f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import classification_report
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
#%%
data["classification"] = lst
data["classification"] = data[["classification"]].shift(-1)
data["classification"] = data["classification"].fillna(0)
#%%

#data = data.drop(labels = {'High','Low','volatility_kch','volatility_kcl','volatility_kcw','volatility_kcp','volatility_kchi',
 #                          'volatility_kcli','volatility_dcl', 'volatility_dch','volume_sma_em','volume_em','volume_vpt',
  #                         'trend_adx','trend_adx_pos','trend_adx_neg','trend_vortex_ind_pos','trend_vortex_ind_neg','trend_vortex_ind_diff',
   #                        'trend_trix','trend_cci','trend_dpo','trend_ichimoku_conv','trend_ichimoku_base','trend_ichimoku_a',
    #                       'trend_ichimoku_b','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_aroon_up','trend_aroon_down',
     #                      'trend_aroon_ind','momentum_uo','momentum_wr','momentum_roc'}, axis=1)
data = data.fillna(0)
data = data.drop(labels = {'High','Low','others_dr','others_dlr','others_cr'}, axis=1)
dataset = data.values
datasets = dataset[42:,:]
#%%
X = datasets[:,:86]

X = X[:,[0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 34, 
         37, 43, 45, 46, 47, 48, 49, 50, 58, 63, 64, 65, 66, 67, 68, 70]]

X_trai = X[:18838]
X_tes = X[18838:]

sc = StandardScaler()
X_trains = sc.fit_transform(X_trai)
X_tests = sc.transform(X_tes)
mm = MinMaxScaler(feature_range=(-1,1))
X_trains = mm.fit_transform(X_trains)
X_tests = mm.transform(X_tests)

#%%

def ylabel(close, steps):
   # print(len(close))
    S = list()
    final_result=0
    #for i in range(len(close)-steps+1):
    for i in range(len(close)):
        # find the end of this pattern
        end_i = i + steps
        #print(end_i)
        # check if we are beyond the dataset
        #final_result=1
        if end_i >= len(close):
            break
        for c in close[i:end_i]:
            #print(c)
            #print(close[end_i])
            #print(len(close[i:end_i]))
            if c >= close[i]*1.05:
                final_result=1
                break
            if c <= close[i]*0.95:
               final_result=0
               break
            #print(final_result)
            #if final_result==1:
            if close[end_i] > close[i]:
               final_result=1
            elif close[end_i] < close[i]:
               final_result=0  
            #print(final_result)
        S.append(final_result)
    #print(S)
    return array(S)

y_trains = ylabel(X[:18839,[1]], 24)
y_test = ylabel(X[18837:,[1]], 24)
#%%
X_t = X_trains[:-24]
y_trains = y_trains[1:]
X_test =X_tests[:-24]
y_test = y_test[1:]
#%% validation
X_train = X_t[:15052]
y_train = y_trains[:15052]
X_vali = X_t[15052:]
y_vali = y_trains[15052:]
#%% Model

#c=26
#gamma = 0.0001
#clf = SVC(C=c, gamma=gamma, kernel="sigmoid")
#clf.fit(X_train, y_train.ravel())

c=10
gamma = 0.001
clf = SVC(C=c, gamma=gamma, kernel="rbf", max_iter=-1)
clf.fit(X_t, y_trains.ravel())
#%%
y_pred = clf.predict(X_vali)

cm = confusion_matrix(y_vali, y_pred)

print(classification_report(y_vali, y_pred))
print(cm)
#%% Test

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

#%%
print(cm)
print(classification_report(y_test, y_pred))
#print('Training set metrics:')
#print('Accuracy:', accuracy_score(y_train, clf.predict(X_train)))
#print('Precision:', precision_score(y_train, clf.predict(X_train)))
#print('Recall:', recall_score(y_train, clf.predict(X_train)))

print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-Score:', f1_score(y_test, y_pred))
#print(cm)

#print(f1_score(y_test, y_pred))
#%%
import seaborn as sns

cm = sns.heatmap(confusion_matrix(y_test, y_pred)/np.sum(confusion_matrix(y_test, y_pred)), annot=True, 
            fmt='.2%', cmap='Blues')

cm.set(title="Confusion Matrix",
      xlabel="Predicted label",
      ylabel="True label",)
#%% Hyperparameter
splits = TimeSeriesSplit(n_splits=5)

#C_range = [40, 50, 60]
#gamma_range = [0.0001, 0.001, 0.01]

#param_grid = dict(gamma=gamma_range, C=C_range)

param_grid = {"C": [1, 10, 20, 30], "kernel": ["rbf", "sigmoid"], "gamma": [0.0001, 0.001, 0.01]} 

grid =RandomizedSearchCV(SVC(), param_grid, scoring = "accuracy", cv=splits, verbose=2, n_jobs=-1)

grid.fit(X_t, y_trains.ravel())

print(grid.best_params_)

#%%Method of feature selection

sfs = SFS(SVC(), k_features=40, forward=True, 
          floating=False, scoring = 'accuracy', verbose=2, cv = splits, n_jobs=-1)

sfs = sfs.fit(X_train, y_train.ravel())

print(sfs.subsets_)

#%% returns

def retur_pro(sign, cl, hours):
    S = list()
    signal = list()
    #len(sign)
    for i in range(0, len(sign), hours):
        end_i = i + hours
        enter = 0
        leave = 0
        # check if we are beyond the dataset
        if end_i >= len(sign):
            #print("A")
            break
        c = sign[i]
        signal.append(c)
        #print("c:" + str(c))
        print("i:" + str(i) + " end_i:" + str(end_i))
        print("INI:" + str(cl[i]) + " FIN:" + str(cl[end_i]))
          
        if c == 0:
            #flag_condition = False
            for e in cl[i+24:end_i+24]:
            #    if e >= cl[i]*1.04:
             #       S.append(-0.04)
              #      flag_condition=True
               #     break
                #if e <= cl[i]*0.95:
                 #   S.append(0.05)
                  #  flag_condition=True
                   # break
            #if not flag_condition:   
                enter = cl[i+24]
                leave = cl[end_i+24]
                r = -(leave-enter)/enter
                S.append(r)
                break
                                        
        if c == 1:
            #flag_condition = False
            for e in cl[i+24:end_i+24]:
             #   if e >= cl[i]*1.05:
              #      S.append(0.05)
               #     flag_condition=True
                #    break
                #if e <= cl[i]*0.96:
                 #   S.append(-0.04)
                  #  flag_condition=True
                   # break
            #if not flag_condition:   
                enter = cl[i+24]
                leave = cl[end_i+24]
                r = (leave-enter)/enter
                S.append(r)
                break
    return array(S)#, array(signal)
                   
results = retur_pro(y_pred[30:], X_tes[7:,[1]], 24) 
#%%
df = pd.DataFrame(results)

filepath = 'SVM_prediction_Buy&Sell_final.xlsx'

df.to_excel(filepath, index=False)
#%% Comulative results

total = 1
hist = []
hist.append(total)
for result in results:
    total = total * (1+ result)
    hist.append(total)
print(total)
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
    new_aloc = port_aloc * 1.1
    if new_aloc > 1:
        return 1
    else:
        return new_aloc
# lose situation
def lose(port_aloc):
    return port_aloc * 0.8

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
    gains = ((X_te[i][1]-X_te[i][0])/X_te[i][0])*order
    cum_sum = cum_sum*(1+gains*port_aloc)
    port_aloc = win_or_lose(gains, port_aloc)
    
print(cum_sum)
#%% Revenue
ret = list()
cl = X_te[:,[1]]
op = X_te[:,[0]]

for i in range(len(y_pred)):  
    gains = ((cl[i]-op[i])/op[i])*y_pred[i]
    ret.append(gains)
    
print(cum_sum)
print(np.mean(ret)*24*365)
#%%
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels = ['Sell', 'Buy']

plot_confusion_matrix(confusion_matrix(y_test, y_pred), labels, title="SVM \n Confusion Matrix")