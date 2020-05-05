# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import glob
import numpy as np
import os
import pandas as pd
import time

# Project files
from src.image_generator import DataGenerator
from src.network import net
from src.utils import getCallbacks


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_images/")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
VALID_X_DIR = TRAIN_X_DIR
VALID_Y_DIR = TRAIN_Y_DIR
TRAIN_VALID_SPLIT = 0.9
TEST_DIR = os.path.join(ROOT, "test_images")

# Model parameters
LOAD_MODEL = False
BATCH_SIZE = 8
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
INPUT_SHAPE = (PATCH_SIZE*4, PATCH_SIZE*4, 3)
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-4
CONCATENATE_PATCHES = True

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
        model = net(INPUT_SHAPE, PATCHES_PER_IMAGE)

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
    train_generator = DataGenerator(
        TRAIN_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
        TRAIN_VALID_SPLIT, concatenate_patches=CONCATENATE_PATCHES)
    train_batch_generator = train_generator.trainImagesAndLabels(
        TRAIN_Y_DIR, normalize=True)
    number_of_train_batches = train_generator.numberOfBatchesPerEpoch()
    valid_generator = DataGenerator(
        VALID_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
        1-TRAIN_VALID_SPLIT, concatenate_patches=CONCATENATE_PATCHES)
    valid_batch_generator = valid_generator.trainImagesAndLabels(
        VALID_Y_DIR, normalize=True)
    number_of_valid_batches = valid_generator.numberOfBatchesPerEpoch()

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
    # Enable multiple GPUs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    train()


main()
