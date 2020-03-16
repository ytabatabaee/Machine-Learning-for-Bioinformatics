import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.stats import t


def read_data(adr, y_name):
    df = pd.read_csv(adr)
    x = df.loc[:, df.columns != y_name]
    y = df.loc[:, df.columns == y_name]
    return x, y


def shuffle(x, y):
    idx = np.random.permutation(x.index)
    x = x.reindex(idx)
    y = y.reindex(idx)
    return x, y


def data_split(x, y, frac):
    idx = np.random.permutation(x.index)
    train_idx = idx[:int(len(idx) * frac)]
    test_idx = idx[int(len(idx) * frac):len(idx)]
    train_x = x.iloc[train_idx, :]
    train_y = y.iloc[train_idx, :]
    test_x = x.iloc[test_idx, :]
    test_y = y.iloc[test_idx, :]
    return train_x, train_y, test_x, test_y


def accuracy(y, y_pred):
    diff = (y == y_pred).to_numpy()
    return np.count_nonzero(diff) / len(y)


def confusion_matrix(y, y_pred):
    tp = np.count_nonzero((y_pred == True) & (y == True).to_numpy())
    fp = np.count_nonzero((y_pred == True) & (y == False).to_numpy())
    tn = np.count_nonzero((y_pred == False) & (y == False).to_numpy())
    fn = np.count_nonzero((y_pred == False) & (y == True).to_numpy())
    return tp, fp, tn, fn


def classification_report(y, y_pred):
    tp, fp, tn, fn = confusion_matrix(y, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1_score = 2 * recall * precision / (recall + precision)
    return accuracy, recall, precision, specificity, f1_score


def paired_t_test(y1, y2, alpha):
    n = len(y1)
    dif = y1 - y2
    mean = np.mean(dif)
    std = np.std(dif)
    t_stat = mean / (std / np.sqrt(n))
    dof = n - 1
    p_value = (1.0 - t.cdf(abs(t_stat), dof)) * 2.0
    print("t-stat = ", t_stat[0])
    print("p-value = ", p_value[0])
    print("aplha = ", alpha)
    if p_value > alpha:
        print("Accept the null hypothesis that the predictions are mostly equal.")
    else:
        print("Reject the null hypothesis that the predictions are mostly equal. There is a meaningful difference between estimators.")
    return


def report_stats(y, y_pred):
    tp, fp, tn, fn = confusion_matrix(y, y_pred)
    print(tabulate([['Predicted Positive', tp, fp], ['Predicted Negative', fn, tn]], headers=['', 'Actually Positive', 'Actually Negative']))
    accuracy_score, recall, precision, specificity, f1_score = classification_report(y, y_pred)
    print("\n\n")
    print(tabulate([[accuracy_score, recall, precision, specificity, f1_score]], headers=['Accuracy', 'Recall', 'Precision', 'Specificity', 'F1 Score']))
