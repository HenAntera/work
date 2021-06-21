# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:15:55 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
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
from collections import deque
from numpy import array
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("database.1.csv", sep=";",header=0, index_col=0)   

statistics = data.describe()

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume",fillna=True)

lst = list()

#forecast = 1
#data["prediction"] = data[["Close"]].shift(-forecast)
#data.drop(data.tail(forecast).index, inplace=True)

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

lolset = lol.values
dataset = data.values
lolset =lolset[42:,:]
dataset = dataset[42:,:]

#X = dataset[:,:79]
X = lolset
y = dataset[:,[82]]

poly = PolynomialFeatures(interaction_only=True)
X = poly.fit_transform(X)

X = X[:,[0, 15, 22, 36, 40, 41, 43, 49, 107, 110, 115, 119, 122, 123, 127, 134, 142, 147, 184, 190, 203, 207, 210, 211, 216, 218,
         220, 221, 222, 224, 227, 230, 232, 255, 258, 264, 288, 295, 299, 302, 303, 309, 318, 334, 357, 379, 387, 388, 390]]


m = np.mean(X, axis=0) 
std = np.std(X, axis=0) 

#X = 0.5 * (np.tanh(0.01 * ((X - m) / std)) + 1)
X=StandardScaler().fit_transform(X)

dataset = np.append(X, y, axis=1)

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

X, y = split_sequences(dataset, 7)

#from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

#dataset = TimeseriesGenerator(
 #   dataset[:,:79],
  #  y,
   # length=7,
   # stride=1,
   # sampling_rate=1,
   # batch_size=128,
   # shuffle=False,
    #)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import datetime

look_back = 7
LOG_DIR = f"{int(time.time())}"

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(50, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=("accuracy"))

def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_units', min_value=32, max_value=256, step=32), input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(Dropout(hp.Float('drop_rate', min_value=0.1, max_value=0.9, step=0.1)))
    model.add(BatchNormalization())
    
    for i in range(hp.Int("n_layers", 0,4)):
        model.add(LSTM(hp.Int('Add_{i}_units', min_value=32, max_value=256, step=32), input_shape=[(X_train.shape[1:])], return_sequences=True))
        model.add(Dropout(hp.Float('drop_rate_1', min_value=0.1, max_value=0.9, step=0.1)))
        model.add(BatchNormalization())

    model.add(LSTM(50, input_shape=(X_train.shape[1:])))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="relu"))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=("accuracy"))

    return model
'''
tuner = RandomSearch(build_model, objective = "val_accuracy", max_trials=2, executions_per_trial = 1, directory = LOG_DIR)

m = tuner.search(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
print(m)
print(best_hps)
'''
NAME = f"{7}-SEQ-{1}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  

checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

num_epochs = 7
model.fit(X_train, y_train.ravel() , epochs=num_epochs, 
          validation_split=0.25, verbose=2, batch_size=64,
          callbacks=[tensorboard_callback])

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#Save model
#model.save("models/{}".format(NAME))
print(model.metrics_names)
