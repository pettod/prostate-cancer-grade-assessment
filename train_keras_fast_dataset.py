# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import glob
import numpy as np
import os
import pandas as pd
import time

# Project files
from src.network import net
from src.utils import getCallbacks, getNumberOfSteps


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "patches_256_4x4")
VALID_X_DIR = TRAIN_X_DIR

# Model parameters
LOAD_MODEL = False
BATCH_SIZE = 16
PATCH_SIZE = 256
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-4

PROGRAM_TIME_STAMP = time.strftime("%Y-%m-%d_%H%M%S")


def loadModel(load_pretrained_model=True, model_root="models"):
    if load_pretrained_model:
        latest_model = sorted(glob.glob(model_root + "/*/*.h5"))[-1]
        model = load_model(
            latest_model,
            custom_objects={
                "tf": tf
            })
        print("Loaded model: {}".format(latest_model))
    else:
        model = net((PATCH_SIZE, PATCH_SIZE, 3))

        # Compile model
        model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
    print("Number of model parameters: {:,}".format(model.count_params()))
    return model


def train():
    # Load model
    model = loadModel(LOAD_MODEL)
    save_root = "models/{}".format(PROGRAM_TIME_STAMP)

    # Load data generators
    train_batch_generator = ImageDataGenerator().flow_from_directory(
        TRAIN_X_DIR, target_size=(PATCH_SIZE, PATCH_SIZE),
        class_mode="categorical", shuffle=True, batch_size=BATCH_SIZE)
    number_of_train_batches = getNumberOfSteps(TRAIN_X_DIR, BATCH_SIZE)
    valid_batch_generator = ImageDataGenerator().flow_from_directory(
        VALID_X_DIR, target_size=(PATCH_SIZE, PATCH_SIZE),
        class_mode="categorical", shuffle=True, batch_size=BATCH_SIZE)
    number_of_valid_batches = getNumberOfSteps(VALID_X_DIR, BATCH_SIZE)

    # Define callbacks
    callbacks = getCallbacks(PATIENCE, save_root, BATCH_SIZE)

    # Start training
    history = model.fit_generator(
        train_batch_generator, steps_per_epoch=number_of_train_batches,
        epochs=EPOCHS, validation_data=valid_batch_generator,
        validation_steps=number_of_valid_batches,
        callbacks=callbacks)
    print("Model saved to: '{}'".format(save_root))


def main():
    # Don't train on GPU 0 (to reduce overheating)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train()


main()
