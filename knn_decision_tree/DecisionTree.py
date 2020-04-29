import pandas as pd
import numpy as np
from scipy import stats


class Node:
    def __init__(self, pred):
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.pred = pred
        self.bound = 0
        self.pos = 0
        self.neg = 0


class DecisionTree:
  def __init__(self, max_depth, threshold):
    self.max_depth = max_depth
    self.threshold = threshold


  def fit(self, x, y):
    self.tree = self.build_tree(x, y, 0)


  def impurity(self, a, b):
    return 1.0 - (a / (a + b)) ** 2 - (b / (a + b)) ** 2


  def split(self, x, y):
    best_feature = None
    bound = None

    split_count = [y.target.value_counts()[0], y.target.value_counts()[1]]
    best_impurity = self.impurity(split_count[0], split_count[1])

    for feature in range(0, x.shape[1]):
        left_split_count = [0, 0]
        right_split_count = split_count.copy()
        x_numpy = x.copy().iloc[:, feature].to_numpy()
        y_numpy = y.copy().iloc[:, 0].to_numpy()
        x_sorted, y_sorted = zip(*sorted(zip(x_numpy, y_numpy)))

        for i in range(1, x.shape[0]):
            target = y_sorted[i - 1]
            left_split_count[target] += 1
            right_split_count[target] -= 1

            if x_sorted[i] == x_sorted[i - 1]:
                continue

            impurity_left = self.impurity(left_split_count[0], left_split_count[1])
            impurity_right = self.impurity(right_split_count[0], right_split_count[1])
            impurity_total = (i * impurity_left + (x.shape[0] - i) * impurity_right) / x.shape[0]

            if impurity_total < best_impurity:
                best_impurity = impurity_total
                best_feature = feature
                bound = (x_sorted[i] + x_sorted[i - 1]) / 2

    return bound, best_feature


  def build_tree(self, x, y, depth):
    node = Node(pred = y.mode().at[0, 'target'])
    node.pos = y.target.sum()
    node.neg = y.size - node.pos

    if y.target.value_counts().size == 1:
        return node

    if depth < self.max_depth and y.target.value_counts()[0] < self.threshold * y.shape[0]:
        bound, feature = self.split(x, y)
        if feature is not None:
            node.feature = feature
            node.bound = bound
            division = x.iloc[:, feature] < bound
            left_x = x[division]
            right_x = x[~division]
            left_y = y[division]
            right_y = y[~division]
            node.left_child = self.build_tree(left_x, left_y, depth + 1)
            node.right_child = self.build_tree(right_x, right_y, depth + 1)

    return node


  def predict(self, test_x):
    test_pred = pd.DataFrame(columns = ['target'], index = test_x.index)

    for i in range(0, len(test_x)):
        node = self.tree
        while node.left_child:
            if test_x.iloc[i, node.feature] < node.bound:
                node = node.left_child
            else:
                node = node.right_child
        test_pred.at[test_x.index[i], 'target'] = node.pred

    return test_pred


def pruning(node):
    left = node.left_child
    right = node.right_child

    if left.left_child:
        left = pruning(left)
    if right.left_child:
        right = pruning(right)

    # both children were leaves
    if not left.left_child and not right.left_child:
        observed = np.array([left.pos, left.neg, right.pos, right.neg])
        s = node.pos + node.neg
        sl = left.pos + left.neg
        sr = right.pos + right.neg
        expected = np.array([sl * node.pos / s, sl * node.neg / s, sr * node.pos / s, sr * node.neg / s])
        if chi_square_prune(0.05, 1, expected, observed):
            node.left_child = None
            node.right_child = None
    return node


def chi_square_prune(alpha, dof, expected, observed):
    z = sum(((expected - observed) ** 2) / expected)
    cv = stats.chi2.ppf(q=1-alpha, df=dof)
    if z > cv:
        return False
    return True
