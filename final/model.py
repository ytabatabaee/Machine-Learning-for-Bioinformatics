# importing libraries

import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def prepare_data(file_path, normalization=False, balance=False, dim_reduction=False):
    # reading data from csv to dataframe
    data = pd.read_csv(file_path)

    # balancing the data
    if balance:
        active = data[data.Outcome=='Active']
        inactive = data[data.Outcome=='Inactive']
        active_balanced = resample(active, replace=True, n_samples=len(inactive), random_state=27)
        data = pd.concat([active_balanced, inactive])

    # building x and y
    X = data.drop(['Outcome'], axis=1).to_numpy()
    y = data['Outcome']
    y = np.where(y.values == 'Active', 1, 0)

    # normalizaition
    if normalization:
        X = normalize(X)

    # dimensionality reduction
    if dim_reduction:
        pca = TruncatedSVD(n_components=20, random_state=42)
        X = pca.fit_transform(X)

    return X, y


# This function takes a csv file as input and returns a trained model as output
def train_assay(file_path):
    X_train, y_train = prepare_data(file_path, normalization=True, balance=True, dim_reduction=False)
    clf = Perceptron()
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_preds = clf_isotonic.predict_proba(X_test)
    return clf_isotonic


# This function takes a model and a test csv file as input and returns the predictions as output
def test_assay(model, file_path):
    X_test, _ = prepare_data(file_path, normalization=True, balance=False, dim_reduction=False)
    predicts = model.predict_proba(X_test)
    return predicts

X_test, y_test = prepare_data('AID604red_test.csv', normalization=True, balance=False, dim_reduction=False)
y_preds = test_assay(train_assay('AID604red_train.csv'), 'AID604red_test.csv')
print("roc score = ", roc_auc_score(y_test, y_preds[:, 1]))
