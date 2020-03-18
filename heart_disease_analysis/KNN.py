import pandas as pd
import numpy as np


class KNN:
  def __init__(self, k):
    self.k = k

  def predict(self, test_x):
    test_x = self.scale_features(test_x)
    missing_cols = set(self.x.columns) - set(test_x.columns)
    for c in missing_cols:
        test_x[c] = 0
    test_pred = pd.DataFrame(columns = ['target'], index = test_x.index)
    dist_matrix = self.euclidean_distance(test_x)

    for i in range(0, len(test_x)):
        neighbor_indexes = dist_matrix[i, :].argsort()[0:self.k]
        majority = self.y.iloc[neighbor_indexes].mode()
        test_pred.at[test_x.index[i], 'target'] = majority.at[0, 'target']

    return test_pred

  def euclidean_distance(self, test_x):
    dist_matrix = np.zeros((test_x.shape[0], self.x.shape[0]))
    dist_matrix = - 2 * np.dot(test_x, self.x.T).T
    dist_matrix += np.diag((np.dot(test_x, test_x.T)))
    dist_matrix = dist_matrix.T
    dist_matrix += np.diag((np.dot(self.x, self.x.T)))
    return np.sqrt(dist_matrix)


  def scale_features(self, x):
    cat_features = ["cp", "restecg", "slope", "thal"]
    x = pd.get_dummies(x, columns=cat_features)
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


  def fit(self, x, y):
    self.x = self.scale_features(x)
    self.y = y
