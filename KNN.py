import pandas as pd
import numpy as np


class KNN:

  def __init__(self, k):
    self.k = k

  def predict(self, test_x):
    test_x = self.scale_features(test_x)
    test_pred = pd.DataFrame(columns = ['target'], index = test_x.index)

    for i in range(0, len(test_x)):
        neighbor_indexes = self.find_neighbors(self.x, self.y, test_x.iloc[i], self.k)
        majority = self.y.iloc[neighbor_indexes].mode()
        test_pred.at[test_x.index[i], 'target'] = majority.at[0, 'target']

    return test_pred


  def euclidean_distance(self, v1, v2):
    dif = v1 - v2
    return np.sqrt(np.sum(dif ** 2))


  def scale_features(self, x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


  def find_neighbors(self, x, y, test_row, k):
    dist = np.zeros(len(x))
    for i in range(0, len(x)):
        dist[i] = self.euclidean_distance(x.iloc[i], test_row)
    indexes = dist.argsort()[0:k]
    return indexes


  def fit(self, x, y):
    self.x = self.scale_features(x)
    self.y = y
