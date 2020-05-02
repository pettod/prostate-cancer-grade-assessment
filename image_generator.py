import numpy as np
import math
import random
import pandas as pd
import os
import tensorflow as tf
import glob
from openslide import OpenSlide


class DataGenerator:
    def __init__(self):
        self.__available_indices = []
        self.__latest_used_indices = []
        self.__image_names = []
        self.__number_of_training_samples = None
        self.__patch_size = None
        self.__train_directory = None
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

    def getImageGenerator(
            self, data_directory, patch_size, normalize=False, shuffle=True):
        self.__shuffle = shuffle
        self.__number_of_training_samples = 0
        for file_name in glob.glob(os.path.join(data_directory, '*')):
            self.__image_names.append(file_name)
            self.__number_of_training_samples += 1

        while True:
            self.__pickIndices()

            # Read images
            images = []
            for i in self.__latest_used_indices:
                image_name = os.path.join(
                    data_directory, self.__image_names[i])
                image_slide = OpenSlide(image_name)
                start_x = 0
                start_y = 0
                images.append(np.asarray(image_slide.read_region((
                    start_x, start_y), 0,
                    (patch_size, patch_size))))
            if normalize:
                images = self.normalizeArray(np.array(images))
            else:
                images = np.array(images)
            yield images

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
        self.__train_directory = image_directory
        self.__batch_size = batch_size
        self.__patch_size = patch_size
        self.__shuffle = shuffle

        labels = pd.read_csv(labels_file_path)["isup_grade"]
        batch_generator = self.getImageGenerator(
            image_directory, patch_size, normalize, shuffle)

        while True:
            X_batch = next(batch_generator)
            y_batch = np.array([labels[i] for i in self.__latest_used_indices])
            y_batch = tf.keras.utils.to_categorical(y_batch, number_of_classes)
            yield X_batch, y_batch
