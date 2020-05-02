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
        image_shape = image_slide.dimensions
        low_resolution_image = np.array(image_slide.read_region((
            0, 0), 2, image_slide.level_dimensions[2]))[..., :3]
        cell_coordinates = np.where(np.mean(
            low_resolution_image, axis=-1) < 200)
        patch_shape = (self.__patch_size, self.__patch_size)
        while True:
            random_coordinate_indices = random.sample(
                range(cell_coordinates[0].shape[0]), 1)
            (start_y, start_x) = (
                cell_coordinates[0][random_coordinate_indices[0]]*16,
                cell_coordinates[1][random_coordinate_indices[0]]*16)
            start_x = min(start_x, image_shape[0] - self.__patch_size - 1)
            start_y = min(start_y, image_shape[1] - self.__patch_size - 1)
            end_x, end_y = np.array([start_x, start_y]) + self.__patch_size
            patch = np.array(image_slide.read_region((
                start_x, start_y), 0, patch_shape))[..., :3]
            if np.mean(patch) < 230:
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
