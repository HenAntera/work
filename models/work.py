import numpy as np
import pandas as pd

df = pd.read_csv("database.1.txt", delimiter=" " )
df = df.values
import ta
from ta.utils import dropna
from ta import add_all_ta_features

# Add all ta features
new_features = add_all_ta_features(
    , open="Open", high="High", low="Low", close="Close", volume="Volume")


regr = AdaBoostRegressor()

regr.fit(X_train,y_train.ravel())

regr.score(X_test,y_test)

y_predict = clf.predict(X_test)

mse = mean_squared_error(y_predict, y_test)
print(mse)

SGDClassifier(), ExtraTreesClassifier(), RandomForestClassifier(), RidgeClassifierCV()