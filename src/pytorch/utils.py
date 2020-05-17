import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def computeMetrics(y, output):
    y_pred = np.argmax(output.cpu().numpy(), axis=1)
    y_true = y.cpu().numpy()
    kappa = cohen_kappa_score(y_pred, y_true, weights="quadratic")
    accuracy = accuracy_score(y_pred, y_true)
    return kappa, accuracy


def computeLoss(y_pred, y_true):
    loss = nn.CrossEntropyLoss()(y_pred, y_true.clone().long()) + \
        0.5 * nn.MSELoss()(torch.argmax(y_pred, dim=1).float(), y_true.float())
    return loss
