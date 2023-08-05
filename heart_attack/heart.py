import os
from numpy.core.fromnumeric import mean
from numpy.core.numeric import cross 
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix
DATASET_PATH = os.path.join("datasets")

def DATASET_LOAD(dataset_path = DATASET_PATH):
    dataset_load = os.path.join(dataset_path, "heart.csv")
    return pd.read_csv(dataset_load)

dataset = DATASET_LOAD()
X = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 13]

X_train, X_test, y_train, y_test = X[:271], X[271:], y[:271], y[271:]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
dt_clf = DecisionTreeClassifier(random_state=1)
rf_clf = RandomForestClassifier(random_state=1)
nb_clf = GaussianNB()
svc_clf = SVC(kernel = 'rbf', random_state=0)

nb_clf.fit(X_train, y_train)
y_pred = nb_clf.predict(X_test)
nb_mse = mean_squared_error(y_test, y_pred)
nb_rmse = np.sqrt(nb_mse)
print(nb_rmse)
print(confusion_matrix(y_test, y_pred))