import tensorflow as tf
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
        model_file_name, verbose=1,
        save_best_only=True, save_weights_only=False)
    reduce_learning_rate = ReduceLROnPlateau(
        factor=0.2, patience=3, min_lr=1e-8)
    csv_logger = CSVLogger(csv_log_file_name, separator=';')
    tensor_board = TensorBoard(
        log_dir=save_root, batch_size=batch_size)
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


def quadraticKappa(
        y_pred, y_true, y_pow=4, eps=1e-10, N=6, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))

        return 1 - (nom / (denom + eps))
