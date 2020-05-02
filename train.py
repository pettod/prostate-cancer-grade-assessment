# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import glob
import os
import pandas as pd
import time

# Project files
from image_generator import DataGenerator
from network import net
from utils import getCallbacks


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_images/")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
VALID_X_DIR = TRAIN_X_DIR
VALID_Y_DIR = TRAIN_Y_DIR
TEST_DIR = os.path.join(ROOT, "test_images")

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


def test():
    submission_df = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    model = loadModel(False)  # Remove False
    test_generator = DataGenerator()
    test_batch_generator = test_generator.getImageGeneratorAndNames(
        TEST_DIR, PATCH_SIZE, BATCH_SIZE, normalize=True, shuffle=False)
    number_of_batches = test_generator.numberOfBatchesPerEpoch(
        TEST_DIR, BATCH_SIZE)
    predictions = []
    for i in range(number_of_batches):
        batch, image_names = next(test_batch_generator)
        predictions.append(model.predict(batch))
    #submission_df = predict_submission(test_df, test_path, passes=3)
    submission_df.to_csv("submission.csv", index=False)
    submission_df.head()


def train():
    # Load model
    model = loadModel(LOAD_MODEL)
    save_root = "models/{}".format(PROGRAM_TIME_STAMP)

    # Load data generators
    train_generator = DataGenerator()
    train_batch_generator = train_generator.trainImagesAndLabels(
        TRAIN_X_DIR, TRAIN_Y_DIR, BATCH_SIZE, PATCH_SIZE, normalize=True)
    number_of_train_batches = train_generator.numberOfBatchesPerEpoch(
        TRAIN_X_DIR, BATCH_SIZE)
    valid_generator = DataGenerator()
    valid_batch_generator = valid_generator.trainImagesAndLabels(
        VALID_X_DIR, VALID_Y_DIR, BATCH_SIZE, PATCH_SIZE, normalize=True)
    number_of_valid_batches = valid_generator.numberOfBatchesPerEpoch(
        VALID_X_DIR, BATCH_SIZE)

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
    train()
    #test()


main()
