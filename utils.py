from tensorflow.keras.callbacks import \
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import os
from sklearn.metrics import confusion_matrix
import numpy as np


def getCallbacks(patience, save_root, batch_size):
    # Define saving file paths
    model_file_name = "{}/model.h5".format(save_root)
    csv_log_file_name = "{}/logs.csv".format(save_root)

    # Create folders if do not exist
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    # Define callbacks
    early_stopping = EarlyStopping(patience=patience)
    checkpointer = ModelCheckpoint(
        model_file_name, verbose=1, save_best_only=True)
    reduce_learning_rate = ReduceLROnPlateau(
        factor=0.3, patience=4, min_lr=1e-8)
    csv_logger = CSVLogger(csv_log_file_name, separator=';')
    tensor_board = TensorBoard(
        log_dir=save_root, write_graph=False, batch_size=batch_size)
    callbacks = [
        early_stopping,
        checkpointer,
        reduce_learning_rate,
        csv_logger,
        tensor_board]
    return callbacks


def quadratic_kappa(actuals, preds, N=5):
    """
    This function calculates the Quadratic Kappa Metric used for Evaluation in
    the PetFinder competition at Kaggle. It returns the Quadratic Weighted
    Kappa metric score between the actual and the predicted values of adoption
    rating.
    """
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)
    E = E/E.sum()
    O = O/O.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j]*O[i][j]
            den += w[i][j]*E[i][j]
    return (1 - (num/den))
