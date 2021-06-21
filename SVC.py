# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:17:29 2020

@author: Henrique Oliveira
"""

import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
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

#data = data.drop(labels={'Close',"others_dr","others_dlr","others_cr",'trend_psar_down','total_volume',
 #                             'trend_aroon_up','momentum_stoch','trend_aroon_ind',
  #                            'momentum_rsi','trend_adx','market_cap'}, axis=1)

lol = data[{'Open','Volume','market_cap','total_volume','Value - Hrate','Value - Uadresses',
            'volume_fi','momentum_mfi','volume_sma_em','volatility_bbm','volatility_kcp','volatility_kcli',
            'volatility_dch','trend_macd_diff','trend_sma_fast','trend_ema_fast','trend_adx','trend_adx_pos',
            'trend_adx_neg','trend_vortex_ind_neg','trend_mass_index','trend_ichimoku_base',
            'trend_psar_up_indicator','momentum_stoch','volatility_bbhi','volatility_bbli','volatility_kchi',
            'trend_psar_down_indicator'}]

dataset = data.values
#lolset = lol.values
dataset = dataset[42:,:]
#lolset =lolset[42:,:]
#X = lolset
X = dataset[:,:79]
X = X[:,[0, 4, 5, 8, 10, 24, 28, 30, 34, 36, 44, 56, 65, 73, 75]] #0.5728813559322035
#X = X[:,[3, 4, 6, 8, 9, 13, 15, 16, 17, 20, 25, 26, 27, 32, 36, 38, 47, 50, 61, 62, 69, 72]] #0.5813559322033898

y = dataset[:,[83]]

poly = PolynomialFeatures(interaction_only=True)
X = poly.fit_transform(X)
#X = X[:,[0, 4, 9, 11, 12, 14, 17, 19, 23, 25, 31, 36, 38, 44, 48, 54, 55, 59, 61, 62, 64, 74, 75, 88, 105, 114]]
#X = X[:,[15, 22, 36, 40, 41, 43, 49, 107, 110, 115, 119, 122, 123, 127, 134, 142, 147, 184, 190, 203, 207, 210, 211, 216, 218,
 #        220, 221, 222, 224, 227, 230, 232, 255, 258, 264, 288, 295, 299, 302, 303, 309, 318, 334, 357, 379, 387, 388, 390]]

#X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, shuffle=False, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#def PolySVC(degree=2, **kwargs):
 #   return make_pipeline(PolynomialFeatures(degree),
  #                       SVC)

c=1

clf = SVC(C=c, degree=1, kernel="rbf")
clf.fit(X_train, y_train.ravel())


y_pred = clf.predict(X_test)

df = pd.DataFrame (y_pred)

filepath = 'sss1s.xlsx'

df.to_excel(filepath, index=False)

cm = confusion_matrix(y_test, y_pred)

print('Training set metrics:')
print('Accuracy:', accuracy_score(y_train, clf.predict(X_train)))
print('Precision:', precision_score(y_train, clf.predict(X_train)))
print('Recall:', recall_score(y_train, clf.predict(X_train)))

print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))

print(cm)

print(f1_score(y_test, y_pred))

#def PolySVC(degree=2, ):
 #   return make_pipeline(PolynomialFeatures(degree),
  #                       SVC()
#c = np.arange(0.1, 20)
#train_score, val_score = validation_curve(SVC(), X, y.ravel(), "C", c, cv=5)

#plt.plot(c, np.median(train_score, 1), color='blue', label='training score')
#plt.plot(c, np.median(val_score, 1), color='red', label='validation score')
#plt.legend(loc='best')
#plt.ylim(0, 1)
#plt.xlabel('C')
#plt.ylabel('score');


'''
def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting 
    the model as we need to find the predicted value for every point in 
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class 
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator
    
    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    return plt

#plt.figure()
#plot_decision_boundaries(X_train, y_train, SVC, C=1, kernel="linear")
#plt.show()
'''
splits = TimeSeriesSplit(n_splits=5)

'''
param_grid = {"C": [1, 5, 10, 25, 50], "kernel": ["linear", "poly", "rbf", "sigmoid"],
              "degree": [1, 2, 3]} #{'C': 50, 'degree': 1, 'kernel': 'sigmoid'}

grid = GridSearchCV(SVC(), param_grid, scoring = "accuracy", cv=splits, verbose=2)

grid.fit(X_train, y_train.ravel())

print(grid.best_params_)
'''

clf = SVC(C=1, degree=1, kernel="rbf")
clf.fit(X_train, y_train.ravel())
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=splits)
print(np.mean(scores))

'''
index = 1
for train_index, valu_index in splits.split(X_train):
	train = X[train_index]
	valu = X[valu_index]
	print('Observations: %d' % (len(train) + len(valu)))
	print('Training Observations: %d' % (len(train)))
	print('Valuation Observations: %d' % (len(valu)))
	
	index += 1
'''