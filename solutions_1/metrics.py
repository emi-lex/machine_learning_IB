import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)
    tp = (y_pred & y_true).sum()
    fp = (y_pred & np.invert(y_true)).sum()
    tn = (np.invert(y_pred) & np.invert(y_true)).sum()
    fn = (np.invert(y_pred) & y_true).sum()

    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-20)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-20)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return (y_pred == y_true).sum() / y_pred.shape[0]


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    sum1 = ((y_true - y_pred) ** 2).sum()
    sum2 = ((y_true - y_pred.mean()) ** 2).sum()
    return 1 - sum1 / sum2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return (((y_true - y_pred) ** 2).sum()) / y_true.shape[0]


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return ((np.abs(y_true - y_pred)).sum()) / y_true.shape[0]
    