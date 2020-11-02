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

data = pd.read_csv("database.csv", sep=";",header=0, index_col=0)   

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

dataset = data.values

dataset = dataset[42:,:]

X = dataset[:,:79]
y = dataset[:,[82]]

m = np.mean(X, axis=0) 
std = np.std(X, axis=0) 

X = 0.5 * (np.tanh(0.01 * ((X - m) / std)) + 1)

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



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

look_back = 7

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=("accuracy"))

NAME = f"{7}-SEQ-{1}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  

checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

num_epochs = 20
model.fit(X_train, y_train.ravel() , epochs=num_epochs, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
#model.save("models/{}".format(NAME))

