# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import datetime
from matplotlib import pyplot
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix

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
#%% Labelling
data["classification"] = lst
data["classification"] = data[["classification"]].shift(-1)
data["classification"] = data["classification"].fillna(0)
#%% Clean
data = data.fillna(0)
data = data.drop(labels = {'High','Low','others_dr','others_dlr','others_cr'}, axis=1)
dataset = data.values
datasets = dataset[42:,:]

#%%
signal_test = np.load("y_predict.npy")

signal_test = np.reshape(signal_test, (4710,1))

signal_train = datasets[:18838,[87]]

signal = np.concatenate((signal_train, signal_test), axis=0)

X = datasets[:,:86]
#%%
X = np.concatenate((X, signal), axis=1)

#%%
aaXX = np.delete(X, [6,7,20,33,34,35,35,41,42,44,55,56,57,58,76,77], axis=1)

X = aaXX[:,[0, 1, 3, 5, 6, 13, 15, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
         35, 36, 38, 39, 40, 41, 42, 44, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 69, 71]]
#%%Split

X_trai = X[:18838]
X_tes = X[18838:]

sc = StandardScaler()
X_trains = sc.fit_transform(X_trai)
X_tests = sc.transform(X_tes)
#%% Label
def split_sequences(sequences, n_steps):
	X = list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x = sequences[i:end_ix, :-1]
		X.append(seq_x)
		
	return array(X)

def ylabel(close, steps):
   # print(len(close))
    S = list()
    #final_result=0
    #for i in range(len(close)-steps+1):
    for i in range(len(close)):
        # find the end of this pattern
        end_i = i + steps
        #print(end_i)
        # check if we are beyond the dataset
        final_result=1
        if end_i >= len(close):
            break
        for c in close[i:end_i]:
            #print(c)
            #print(len(close[i:end_i]))
            if c >= close[i]*1.10:
                final_result=2
                break
            if c <= close[i]*0.95:
               final_result=0
               break
            #print(final_result)
            if final_result==1:
                
                if close[end_i] > close[i]*1.02:
                    final_result=2
                elif close[end_i] < close[i]*0.98:
                    final_result=0  
            #print(final_result)
        S.append(final_result)
    #print(S)
    return array(S)

X_train = split_sequences(X_trains, 24)
X_test = split_sequences(X_tests,24)
y_training = ylabel(X_trai[:,[1]], 23)
y_testing = ylabel(X_tes[:,[1]], 23)

#X_vali = X_train[14111:]
#y_vali = y_train[14111:]

#y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
#y_test = tf.keras.utils.to_categorical(y_testing, num_classes=3)
#y_vali = tf.keras.utils.to_categorical(y_vali, num_classes=3)
#%%Processing

#sel = VarianceThreshold(threshold=(0.8 * (1 - .8)))
#X_train = sel.fit_transform(X_tra)
#X_test =sel.transform(X_te)

#sc = StandardScaler()
#X_train = sc.fit_transform(X_tra)
#X_test = sc.transform(X_te)
#%% Model
#look_back = 20
LOG_DIR = f"{int(time.time())}"

model = Sequential()
model.add(LSTM(224, input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(LSTM(96, input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(96, input_shape=(X_train.shape[1:]), activation="relu"))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(10, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(3, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=("accuracy"))
#%% Metrics
NAME = f"{7}-SEQ-best-{1}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

file = "RNN_Final_best"  

cp = tf.keras.callbacks.ModelCheckpoint(filepath=file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

log_dir = "logs/fit25/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
#es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
cb_list = [tb, cp]

num_epochs = 30
o = model.fit(X_train, y_training, epochs=num_epochs, 
          validation_split=0.25, verbose=2, batch_size=32,
          callbacks=cb_list)

#score = model.evaluate(X_test, y_testing, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#Test loss: 0.1902562379837036
#Test accuracy: 0.9248986840248108
#%% Load best model

saved_model = load_model(file)
score = saved_model.evaluate(X_test, y_testing, verbose=1)
#%% Evaluate
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
# plot training history
pyplot.plot(o.history['loss'], label='train')
pyplot.plot(o.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
#%% Predict and metrics
yyy = saved_model.predict_classes(X_test)

#y_test = tf.keras.utils.to_categorical(y_testing, num_classes=3)
#y_pred = tf.keras.utils.to_categorical(yyy, num_classes=3)
multi_class_cm = confusion_matrix(y_testing, yyy)
f_micro = f1_score(y_testing, yyy, average='micro')
f_macro = f1_score(y_testing, yyy, average='macro')
f_weight = f1_score(y_testing, yyy, average='weighted')

print('f1 - micro', f_micro)
print('f1 - macro', f_macro)
print('f1 - weighted', f_weight)
#%%save

df = pd.DataFrame(yyy)

filepath = 'lstm_prediction_best_1.xlsx'

df.to_excel(filepath, index=False)
#%% Hyper Tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_units', min_value=32, max_value=256, step=32), input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
    model.add(Dropout(hp.Float('drop_rate', min_value=0.1, max_value=0.9, step=0.1)))
    model.add(BatchNormalization())
    
    for i in range(hp.Int("n_layers", 0,4)):
        model.add(LSTM(hp.Int('Add_{i}_units', min_value=32, max_value=256, step=32), input_shape=[(X_train.shape[1:])], return_sequences=True, activation="relu"))
        model.add(Dropout(hp.Float('drop_rate_1', min_value=0.1, max_value=0.9, step=0.1)))
        model.add(BatchNormalization())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation="softmax"))

    opt = tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]), decay=1e-6)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=("accuracy"))

    return model

#tuner = RandomSearch(build_model, objective = "val_accuracy", max_trials=2, executions_per_trial = 1, directory = LOG_DIR)

#m = tuner.search(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_vali, y_vali))
#best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
#print(m)
#print(best_hps)



''' 0.90 result
#%% Model
look_back = 20
LOG_DIR = f"{int(time.time())}"

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(50, input_shape=(X_train.shape[1:]), activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(3, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=("accuracy"))
'''
#%%

y_prob = saved_model.predict_proba(X_test)
#%%

def retur_pro(sign, cl, hours):
    S = list()
    
    for i in range(0, len(sign), hours):
        end_i = i + hours
        enter = 0
        leave = 0
        # check if we are beyond the dataset
        if end_i > len(sign):
            print("A")
            break
        c = sign[i]
        #print("c:" + str(c))
        print("i:" + str(i) + " end_i:" + str(end_i))
        print("INI:" + str(cl[i]) + " FIN:" + str(cl[end_i]))
          
        if c == 1:
            S.append(0)
            
        if c == 0:
            flag_condition = False
            for e in cl[i:end_i]:
                if e >= cl[i]*1.05:
                    S.append(-0.05)
                    flag_condition=True
                    break
                if e <= cl[i]*0.9:
                    S.append(0.1)
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
                if e >= cl[i]*1.10:
                    S.append(0.1)
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
                   
results = retur_pro(yyy[7:], X_tes[7:,[1]], 24) 
 #%% Allocation           
            
prob =  y_prob[7:]
array=[]
for i in range(0, len(prob), 24):
    
    end_i = i + 24
    if end_i > len(prob):
        break
    
    if prob[i]>0.9:
        array.append(1)
    elif prob[i]>0.8:
        array.append(0.8)
    elif prob[i]>0.7:
        array.append(0.6)
    elif prob[i]>0.6:
        array.append(0.4)
    elif prob[i]>0.5:
        array.append(0.2)        
    elif prob[i]>0.33:
        array.append(0.1)
    
            
            
            
            
            
            
            
            
            
            
            
            