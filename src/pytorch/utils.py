import torch
import torch.nn as nn

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


def plotLearningCurve(log_file_path=None, model_root="models/pytorch"):
    # Read CSV log file
    if log_file_path is None:
        log_file_path = sorted(glob.glob(os.path.join(
            model_root, *['*', "*.csv"])))[-1]
    log_file = pd.read_csv(log_file_path)

    # Read data into dictionary
    log_data = {}
    for column in log_file:
        if column == "epoch":
            log_data[column] = np.array(log_file[column].values, dtype=np.str)
        elif column == "learning_rate":
            continue
        else:
            log_data[column] = np.array(log_file[column].values)

    # Remove extra printings of same epoch
    epoch_string_data = []
    previous_epoch = -1
    for epoch in reversed(log_data["epoch"]):
        if epoch != previous_epoch:
            epoch_string_data.append(epoch)
        else:
            epoch_string_data.append('')
        previous_epoch = epoch
    epoch_string_data = epoch_string_data[::-1]
    number_of_rows = len(epoch_string_data)
    log_data.pop("epoch", None)

    # Define train and validation subplots
    figure_dict = {}
    for key in log_data.keys():
        metric = key.split('_')[-1]
        if metric not in figure_dict:
            figure_dict[metric] = len(figure_dict.keys()) + 1
    number_of_subplots = len(figure_dict.keys())

    # Plot learning curves
    plt.figure(figsize=(15, 7))
    for i, key in enumerate(log_data.keys()):
        metric = key.split('_')[-1]
        plt.subplot(1, number_of_subplots, figure_dict[metric])
        plt.plot(range(number_of_rows), log_data[key], label=key)
        plt.xticks(range(number_of_rows), epoch_string_data)
        plt.xlabel("Epoch")
        plt.title(metric.title())
        plt.legend()
    plt.tight_layout()
    plt.savefig("{}.{}".format(log_file_path.split('.')[0], "png"))
