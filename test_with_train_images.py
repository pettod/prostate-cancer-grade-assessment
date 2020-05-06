# Libraries
from tensorflow.keras.models import load_model
import tensorflow as tf

import glob
import time
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Project files
from src.image_generator import DataGenerator


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "patches_256_4x4")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")

# Model parameters
BATCH_SIZE = 16
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
CONCATENATE_PATCHES = True
NUMBER_OF_COMPUTED_BATCHES = 20


def loadModel(load_pretrained_model=True, model_root="models"):
    latest_model = sorted(glob.glob(model_root + "/*/*.h5"))[-1]
    model = load_model(
        latest_model,
        custom_objects={
            "tf": tf
        })
    print("Loaded model: {}".format(latest_model))
    print("Number of model parameters: {:,}".format(model.count_params()))
    return model


def test():
    # Define submission file, model and data generator
    model = loadModel()
    train_generator = DataGenerator(
        TRAIN_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
        concatenate_patches=CONCATENATE_PATCHES, normalize=True, shuffle=False)
    train_batch_generator = train_generator.trainImagesAndLabels(TRAIN_Y_DIR)
    number_of_train_batches = train_generator.numberOfBatchesPerEpoch()

    # Get image names and predictions
    y_preds = []
    y_trues = []
    for i in range(number_of_train_batches):
        X_batch, y_batch = next(train_batch_generator)
        y_batch = np.argmax(y_batch, axis=1)
        batch_predictions = model.predict(X_batch.astype(np.float32))
        batch_classes = np.argmax(batch_predictions, axis=1)
        batch_accuracy = accuracy_score(y_batch, batch_classes)
        y_preds += list(batch_classes)
        y_trues += list(y_batch)
        print("Batch {}/{}. Accuracy: {}           ".format(
            i+1, number_of_train_batches, batch_accuracy), end="\r")
    accuracy = accuracy_score(y_preds, y_trues)
    kappa = cohen_kappa_score(y_preds, y_trues)
    print("Kappa: {}. Accuracy: {}. Number of images: {}".format(
        kappa, accuracy, len(y_preds)))


def main():
    # Enable multiple GPUs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Test
    test()


main()
