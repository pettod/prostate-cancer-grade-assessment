# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import glob
import math
import numpy as np
import os
import pandas as pd
import time

# Project files
from src.keras.network import net
from src.keras.utils import getCallbacks, quadraticKappa
from src.image_generator import DataGenerator


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_patches_256_4x4_low_res")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
VALID_X_DIR = os.path.join(ROOT, "valid_patches_256_4x4_low_res")
VALID_Y_DIR = os.path.join(ROOT, "valid.csv")

# Model parameters
LOAD_MODEL = True
BATCH_SIZE = 32
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-4

PROGRAM_TIME_STAMP = time.strftime("%Y-%m-%d_%H%M%S")


def loadModel(load_pretrained_model=True, model_root="models/keras"):
    if load_pretrained_model:
        latest_model = sorted(glob.glob(model_root + "/*/*.h5"))[-1]
        model = load_model(
            latest_model,
            custom_objects={
                "tf": tf,
                "quadraticKappa": quadraticKappa,
            })
        print("Loaded model: {}".format(latest_model))
    else:
        input_side = PATCH_SIZE * int(math.sqrt(PATCHES_PER_IMAGE))
        input_shape = (input_side, input_side, 3)
        model = net(input_shape)

        # Compile model
        model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy", quadraticKappa])
    print("Number of model parameters: {:,}".format(model.count_params()))
    return model


def train():
    # Load model
    model = loadModel(LOAD_MODEL)
    save_root = "models/keras/{}".format(PROGRAM_TIME_STAMP)

    # Load data generators
    train_generator = DataGenerator(
        TRAIN_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE, normalize=True,
        rotate=True)
    train_batch_generator = train_generator.trainImagesAndLabels(TRAIN_Y_DIR)
    number_of_train_batches = train_generator.numberOfBatchesPerEpoch()
    valid_generator = DataGenerator(
        VALID_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE, normalize=True)
    valid_batch_generator = valid_generator.trainImagesAndLabels(VALID_Y_DIR)
    number_of_valid_batches = valid_generator.numberOfBatchesPerEpoch()

    # Define callbacks
    callbacks = getCallbacks(PATIENCE, save_root, BATCH_SIZE)

    # Start training
    history = model.fit_generator(
        train_batch_generator,
        epochs=EPOCHS,
        steps_per_epoch=number_of_train_batches,
        validation_data=valid_batch_generator,
        validation_steps=number_of_valid_batches,
        callbacks=callbacks)
    print("Model saved to: '{}'".format(save_root))


def main():
    if False:
        # Enable multiple GPUs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        # Don't train on GPU 0 (to reduce overheating)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train()


main()
