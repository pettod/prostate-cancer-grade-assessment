from tensorflow.keras.callbacks import \
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import os
import math
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
        model_file_name, verbose=1, save_best_only=True,
        save_weights_only=False)
    reduce_learning_rate = ReduceLROnPlateau(
        factor=0.3, patience=5, min_lr=1e-8)
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


def getNumberOfSteps(data_directory, batch_size):
    return math.floor(sum(
        [len(files) for r, d, files in os.walk(data_directory)]) / batch_size)
