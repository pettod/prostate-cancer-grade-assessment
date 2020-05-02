# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import \
    Activation, Add, BatchNormalization, Conv2D, Input, Lambda, UpSampling2D, \
    Reshape, concatenate, Conv2DTranspose, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import \
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import glob
import time
import numpy as np
import math
import random
import pandas as pd
import os
from openslide import OpenSlide


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


class DataGenerator:
    def __init__(self):
        self.__available_indices = []
        self.__latest_used_indices = []
        self.__image_names = []
        self.__number_of_training_samples = None
        self.__patch_size = None
        self.__batch_size = None
        self.__shuffle = None

    def __pickIndices(self):
        # Define indices
        if len(self.__available_indices) == 0:
            self.__available_indices = list(
                np.arange(0, self.__number_of_training_samples))
        if self.__batch_size < len(self.__available_indices):
            if self.__shuffle:
                random_indices_from_list = random.sample(
                    range(len(self.__available_indices)), self.__batch_size)
                self.__latest_used_indices = []
                for i in random_indices_from_list:
                    self.__latest_used_indices.append(
                        self.__available_indices[i])
            else:
                self.__latest_used_indices = self.__available_indices[
                    :self.__batch_size].copy()
        else:
            self.__latest_used_indices = self.__available_indices.copy()

        # Remove indices from availabe indices
        for i in reversed(sorted(self.__latest_used_indices)):
            self.__available_indices.remove(i)

    def __cropPatch(self, image_name):
        image_slide = OpenSlide(image_name)
        patch_shape = (self.__patch_size, self.__patch_size)
        start_x, start_y = 0, 0
        patch = np.asarray(image_slide.read_region((
            start_x, start_y), 0, patch_shape))[..., :3]
        return patch

    def getImageGeneratorAndNames(
            self, data_directory, patch_size, batch_size, normalize=False,
            shuffle=True):
        self.__shuffle = shuffle
        self.__number_of_training_samples = 0
        self.__patch_size = patch_size
        self.__batch_size = batch_size
        for file_name in sorted(glob.glob(os.path.join(data_directory, '*'))):
            self.__image_names.append(file_name)
            self.__number_of_training_samples += 1

        while True:
            self.__pickIndices()

            # Read images
            image_names = [
                self.__image_names[i] for i in self.__latest_used_indices]
            images = np.array([
                self.__cropPatch(os.path.join(
                    data_directory, self.__image_names[i]))
                for i in self.__latest_used_indices])
            if normalize:
                images = self.normalizeArray(np.array(images))
            yield images, image_names

    def normalizeArray(self, data_array, max_value=255):
        return ((data_array / max_value - 0.5) * 2).astype(np.float32)

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array.astype(np.uint8)

    def numberOfBatchesPerEpoch(
            self, data_directory, batch_size):
        number_of_dataset_images = 0
        for file_name in glob.glob(os.path.join(data_directory, '*')):
            number_of_dataset_images += 1
        return math.ceil(number_of_dataset_images / batch_size)

    def trainImagesAndLabels(
            self, image_directory, labels_file_path, batch_size,
            patch_size, normalize=False, shuffle=True,
            number_of_classes=6):
        self.__batch_size = batch_size
        self.__patch_size = patch_size
        self.__shuffle = shuffle

        labels = pd.read_csv(labels_file_path)["isup_grade"]
        batch_generator = self.getImageGeneratorAndNames(
            image_directory, patch_size, batch_size, normalize, shuffle)

        while True:
            X_batch, image_names = next(batch_generator)
            y_batch = np.array([labels[i] for i in self.__latest_used_indices])
            y_batch = tf.keras.utils.to_categorical(y_batch, number_of_classes)
            yield X_batch, y_batch


def net(input_shape):
    h, w, c = input_shape
    inputs = Input(shape=input_shape)
    x = inputs
    kernels = createKernels()
    for kernel in kernels:
        x = kernel(x)
    return Model(inputs=inputs, outputs=x)


def createKernels():
    kernels = []
    kernels.append(Flatten())
    kernels.append(Dense(6, activation="softmax", use_bias=False))
    return kernels


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
    # Define submission file, model and data generator
    submission_file = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    model = loadModel(False)  # Remove False
    test_generator = DataGenerator()
    test_batch_generator = test_generator.getImageGeneratorAndNames(
        TEST_DIR, PATCH_SIZE, BATCH_SIZE, normalize=True, shuffle=False)
    number_of_batches = test_generator.numberOfBatchesPerEpoch(
        TEST_DIR, BATCH_SIZE)

    # Get image names and predictions
    predictions = []
    image_names = []
    for i in range(number_of_batches):
        batch, batch_image_names = next(test_batch_generator)
        image_names += batch_image_names
        predictions += list(np.argmax(model.predict(batch), axis=1))

    # Write submission file
    for i in range(len(predictions)):
        submission_file.at[i, "image_id"] = \
            image_names[i].split('/')[-1].split('.')[0]
        submission_file.at[i, "isup_grade"] = predictions[i]
    submission_file.to_csv("submission.csv", index=False)
    submission_file.head()


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
    #train()
    test()


main()
