import numpy as np
import math
import random
import pandas as pd
import os
import tensorflow as tf
import glob
from openslide import OpenSlide
import cv2


class DataGenerator:
    def __init__(self):
        self.__available_indices = []
        self.__latest_used_indices = []
        self.__image_names = []
        self.__number_of_training_samples = None
        self.__patch_size = None
        self.__batch_size = None
        self.__shuffle = None
        self.__patches_per_image = None

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

    def __cropPatches(self, image_name, downsample_level=0):
        # downsample_level : 0, 1, 2
        # NOTE: only level 0 seems to work currently, other levels crop white
        # areas
        image_slide = OpenSlide(image_name)

        # Resolution downsample levels: 1, 4, 16
        resolution_relation = 4 ** (2 - downsample_level)
        image_shape = image_slide.level_dimensions[downsample_level]

        # Find coordinates from where to select patch
        low_resolution_image = np.array(image_slide.read_region((
            0, 0), 2, image_slide.level_dimensions[2]))[..., :3]
        cell_coordinates = np.array(np.where(np.mean(
            low_resolution_image, axis=-1) < 200)) - \
            int(self.__patch_size / 2 / resolution_relation)
        cell_coordinates[cell_coordinates < 0] = 0
        if cell_coordinates.shape[1] == 0:
            random_coordinates = []
            for i in range(100):
                random_x = random.randint(
                    0, image_shape[0] - self.__patch_size - 1)
                random_y = random.randint(
                    0, image_shape[1] - self.__patch_size - 1)
                random_coordinates.append([random_y, random_x])
            cell_coordinates = np.transpose(np.array(random_coordinates))

        # Crop patches
        patches = []
        for i in range(self.__patches_per_image):
            j = 0
            while True:
                j += 1
                random_index = random.randint(0, cell_coordinates.shape[1] - 1)

                # Scale coordinates by the number of resolution relation
                # between low-resolution image and high/mid-resolution
                start_y, start_x = \
                    cell_coordinates[:, random_index] * resolution_relation
                start_x = min(
                    start_x, image_shape[0] - self.__patch_size - 1)
                start_y = min(
                    start_y, image_shape[1] - self.__patch_size - 1)
                end_x, end_y = np.array(
                    [start_x, start_y]) + self.__patch_size

                # Crop from mid/high resolution image
                patch = np.array(image_slide.read_region((
                    start_x, start_y), downsample_level,
                    (self.__patch_size, self.__patch_size)))[..., :3]

                # Patch has enough colored areas (not pure white) or has been
                # iterated more than 5 times
                if np.mean(patch) < 230 or j >= 5:
                    patches.append(patch)
                    break
        return patches

    def getImageGeneratorAndNames(
            self, data_directory, batch_size, patch_size,
            patches_per_image=1, normalize=False, shuffle=True):
        self.__shuffle = shuffle
        self.__number_of_training_samples = 0
        self.__patch_size = patch_size
        self.__batch_size = batch_size
        self.__patches_per_image = patches_per_image
        for file_name in sorted(glob.glob(os.path.join(data_directory, '*'))):
            self.__image_names.append(file_name)
            self.__number_of_training_samples += 1

        while True:
            self.__pickIndices()

            # Read images
            image_names = [
                self.__image_names[i] for i in self.__latest_used_indices]
            images = np.moveaxis(np.array([
                self.__cropPatches(os.path.join(
                    data_directory, self.__image_names[i]))
                for i in self.__latest_used_indices]), 0, 1)
            if normalize:
                images = self.normalizeArray(np.array(images))
            yield list(images), image_names

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
            patch_size, patches_per_image=1, normalize=False, shuffle=True,
            number_of_classes=6):
        labels = pd.read_csv(labels_file_path)["isup_grade"]
        batch_generator = self.getImageGeneratorAndNames(
            image_directory, batch_size, patch_size, patches_per_image,
            normalize, shuffle)

        while True:
            X_batch, image_names = next(batch_generator)
            y_batch = np.array([labels[i] for i in self.__latest_used_indices])
            y_batch = tf.keras.utils.to_categorical(y_batch, number_of_classes)
            yield X_batch, y_batch
