# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:43:48 2020

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import GenericUnivariateSelect, f_classif

data = pd.read_csv("database.csv", sep=";",header=0, index_col=0)   

statistics = data.describe()

data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume",fillna=True)

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

lol = data[{'Open','Volume','market_cap','total_volume','Value - Hrate','Value - Uadresses',
            'volume_fi','momentum_mfi','volume_sma_em','volatility_bbm','volatility_kcp','volatility_kcli',
            'volatility_dch','trend_macd_diff','trend_sma_fast','trend_ema_fast','trend_adx','trend_adx_pos',
            'trend_adx_neg','trend_vortex_ind_neg','trend_mass_index','trend_ichimoku_base',
            'trend_psar_up_indicator','momentum_stoch','volatility_bbhi','volatility_bbli','volatility_kchi',
            'trend_psar_down_indicator'}]

#'Low','volume_nvi','volatility_atr','volatility_bbp',
 #           'volatility_bbli','trend_ema_slow','trend_vortex_ind_pos','trend_vortex_ind_diff',
  #          'trend_ichimoku_conv','trend_ichimoku_b','trend_visual_ichimoku_b','momentum_roc'}

ones = data[{'volatility_bbhi','volatility_bbli','volatility_kchi',
            'trend_psar_down_indicator','trend_psar_up_indicator','volatility_kcli'}]

dataset = data.values 
features = lol.values

dataset = dataset[42:,:]

features = features[42:,:]

#X = features
X = dataset[:,:79]
y = dataset[:,[82]]

#poly = PolynomialFeatures(interaction_only=True)
#X = poly.fit_transform(X)
#X = X[:,[0, 15, 22, 36, 40, 41, 43, 49, 107, 110, 115, 119, 122, 123, 127, 134, 142, 147, 184, 190, 203, 207, 210, 211, 216, 218,
#         220, 221, 222, 224, 227, 230, 232, 255, 258, 264, 288, 295, 299, 302, 303, 309, 318, 334, 357, 379, 387, 388, 390]]

#[0, 15, 22, 36, 40, 41, 43, 49, 107, 110, 115, 119, 122, 123, 127, 134, 142, 147, 184, 190, 203, 207, 210, 211, 216, 218,
#         220, 221, 222, 224, 227, 230, 232, 255, 258, 264, 288, 295, 299, 302, 303, 309, 318, 334, 357, 379, 387, 388, 390]

#X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

'''VARIANCE'''
#sel = VarianceThreshold(threshold=(0.8 * (1 - .8)))
#X_train = sel.fit_transform(X_train)
#transformer = GenericUnivariateSelect(f_classif, mode='k_best', param=23)
#X_train = transformer.fit_transform(X_train, y_train)

from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator = ExtraTreesClassifier(), step=1, cv=5, scoring='accuracy')

rfecv.fit(X_train, y_train.ravel())

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

'''
#aaa = np.var(X_train, axis=0)

#o = np.corrcoef(X_train, rowvar=False)


n_estim = 50
a = 0.01

#clf = AdaBoostClassifier(n_estimators=n_estim, random_state=0, learning_rate=a)
#clf.fit(X_train, y_train.ravel())
#scores = cross_val_score(clf, X, y.ravel(), cv=7, scoring="accuracy")
#print(np.mean(scores))

c=10

sss = SVC(C=c, degree=1, kernel="linear")
sss.fit(X_train, y_train.ravel())
score_svm = cross_val_score(sss, X, y.ravel(), cv=7)
print(np.mean(score_svm))

#rf=RandomForestClassifier(n_estimators=100)
#rf.fit(X_train, y_train.ravel())

print('Training set metrics:')
print('Accuracy:', accuracy_score(y_train, sss.predict(X_train)))
print('Precision:', precision_score(y_train, sss.predict(X_train)))
print('Recall:', recall_score(y_train, sss.predict(X_train)))

print('Test set metrics:')
print('Accuracy:', accuracy_score(y_test, sss.predict(X_test)))
print('Precision:', precision_score(y_test, sss.predict(X_test)))
print('Recall:', recall_score(y_test, sss.predict(X_test)))

#col_sorted_by_importance=rf.feature_importances_.argsort()

#print(rf.feature_importances_)

#X = data.iloc[:,:79]

#feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
#feat_importances.nlargest(35).plot(kind='barh')
#plt.show()

#print(feat_importances.nlargest(35))

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#sfs = SFS(SVC(), k_features=60, forward=False, 
        #  floating=False, scoring = 'accuracy', verbose=2, cv = 6, n_jobs=-1)

#sfs = sfs.fit(X_train, y_train.ravel())

#print(sfs.subsets_)

#Method of feature selection

#sfs = SFS(SVC(), k_features=50, forward=True, 
 #         floating=False, scoring = 'accuracy', verbose=2, cv = 6, n_jobs=-1)

#sfs = sfs.fit(X_train, y_train.ravel())

#print(sfs.subsets_)
'''

