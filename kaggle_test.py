# Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import glob
import time
import numpy as np
import math
import random
import pandas as pd
import os
import cv2
from PIL import Image
from openslide import OpenSlide


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TEST_DIR = os.path.join(ROOT, "test_images")

# Model parameters
BATCH_SIZE = 8
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
CONCATENATE_PATCHES = True


class DataGenerator:
    def __init__(
            self, data_directory, batch_size, patch_size, patches_per_image=1,
            concatenate_patches=False, normalize=False, shuffle=True,
            rotate=False):
        self.__available_indices = []
        self.__batch_size = batch_size
        self.__concatenate_patches = concatenate_patches
        self.__data_directory = data_directory
        self.__data_stored_into_folders = None
        self.__image_names = []
        self.__labels = None
        self.__latest_used_indices = []
        self.__normalize = normalize
        self.__patch_size = patch_size
        self.__patches_per_image = patches_per_image
        self.__rotate = rotate
        self.__shuffle = shuffle

        self.__defineFileNames()

    def __pickIndices(self):
        # Define indices
        if len(self.__available_indices) == 0:
            self.__available_indices = list(
                np.arange(0, self.__image_names.shape[0]))
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

    def __getCellCoordinatesFromImage(
            self, image_slide, resolution_relation, image_shape):

        # Read low resolution image (3 images resolutions)
        low_resolution_image = np.array(image_slide.read_region((
            0, 0), 2, image_slide.level_dimensions[2]))[..., :3]

        # Find pixels which have cell / exclude white pixels
        # Take center of the cell coordinate by subtracting 0.5*patch_size
        cell_coordinates = np.array(np.where(np.mean(
            low_resolution_image, axis=-1) < 200)) - \
            int(self.__patch_size / 2 / resolution_relation)
        cell_coordinates[cell_coordinates < 0] = 0

        # If image includes only white areas or very white, generate random
        # coordinates
        if cell_coordinates.shape[1] == 0:
            random_coordinates = []
            for i in range(100):
                random_x = random.randint(
                    0, image_shape[0] - self.__patch_size - 1)
                random_y = random.randint(
                    0, image_shape[1] - self.__patch_size - 1)
                random_coordinates.append([random_y, random_x])
            cell_coordinates = np.transpose(np.array(random_coordinates))
        return cell_coordinates

    def __cropPatchesFromImage(self, image_name, downsample_level=0):
        # downsample_level : 0, 1, 2
        # NOTE: only level 0 seems to work currently, other levels crop white
        # areas
        image_slide = OpenSlide(image_name)

        # Resolution downsample levels: 1, 4, 16
        resolution_relation = 4 ** (2 - downsample_level)
        image_shape = image_slide.level_dimensions[downsample_level]

        # Find coordinates from where to select patch
        cell_coordinates = self.__getCellCoordinatesFromImage(
            image_slide, resolution_relation, image_shape)

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

    def __cropPatchesFromImages(self):
        images = []
        for i in self.__latest_used_indices:
            images.append(self.__cropPatchesFromImage(self.__image_names[i]))
        return np.moveaxis(np.array(images), 0, 1)

    def __defineFileNames(self):
        file_names = sorted(glob.glob(os.path.join(
            self.__data_directory, '*')))

        # Data has been stored into class folders
        if len(file_names) > 0 and len(file_names[0].split('.')) == 1:
            self.__data_stored_into_folders = True
            file_names = sorted(glob.glob(os.path.join(
                self.__data_directory, *['*', '*'])))
        else:
            self.__data_stored_into_folders = False
        self.__image_names = np.array(file_names)

    def __getLabels(self, number_of_classes):
        if self.__data_stored_into_folders:
            y_batch = np.array([
                self.__image_names[i].split('/')[-2]
                for i in self.__latest_used_indices])
        else:
            y_batch = np.array(
                [self.__labels[i] for i in self.__latest_used_indices])
        return tf.keras.utils.to_categorical(y_batch, number_of_classes)

    def __createSquarePatches(self, batch):
        batch = list(np.moveaxis(batch, 0, 1))
        patches_per_row = int(math.sqrt(self.__patches_per_image))
        concat_batch = []
        for patches in batch:
            hconcat_patches = []
            for i in range(patches_per_row):
                hconcat_patches.append(cv2.hconcat(patches[:patches_per_row]))
                patches = patches[patches_per_row:]
            concat_batch.append(cv2.vconcat(hconcat_patches))
        return np.array(concat_batch)

    def __readPngImages(self):
        images = []
        for i in self.__latest_used_indices:
            images.append(np.array(Image.open(self.__image_names[i])))
        return np.array(images)

    def __rotateBatchImages(self, images):
        rotated_images = []
        for i in range(images.shape[0]):
            random_angle = random.randint(0, 3)
            rotated_images.append(np.rot90(images[i], random_angle))
        return np.array(rotated_images)

    def getImageGeneratorAndNames(self):
        while True:
            self.__pickIndices()

            # Read images
            image_names = [
                self.__image_names[i] for i in self.__latest_used_indices]
            if image_names[0].split('.')[-1] == "tiff":
                images = self.__cropPatchesFromImages()
                if self.__concatenate_patches:
                    images = self.__createSquarePatches(images)
            else:
                images = self.__readPngImages()
            if self.__rotate:
                images = self.__rotateBatchImages(images)
            if self.__normalize:
                images = self.normalizeArray(images)
            yield images, image_names

    def normalizeArray(self, data_array, max_value=255):
        return ((data_array / max_value - 0.5) * 2).astype(np.float32)

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array.astype(np.uint8)

    def numberOfBatchesPerEpoch(self):
        return math.ceil(self.__image_names.shape[0] / self.__batch_size)

    def trainImagesAndLabels(
            self, labels_file_path=None, number_of_classes=6):
        if not self.__data_stored_into_folders:
            self.__labels = pd.read_csv(
                labels_file_path)["isup_grade"].values.tolist()
        batch_generator = self.getImageGeneratorAndNames()

        while True:
            X_batch, image_names = next(batch_generator)
            y_batch = self.__getLabels(number_of_classes)
            yield X_batch, y_batch


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
    submission_file = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    model = loadModel()
    test_generator = DataGenerator(
        TEST_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
        concatenate_patches=CONCATENATE_PATCHES, normalize=True, shuffle=False)
    test_batch_generator = test_generator.getImageGeneratorAndNames()
    number_of_batches = test_generator.numberOfBatchesPerEpoch()

    # Get image names and predictions
    predictions = []
    image_names = []
    for i in range(number_of_batches):
        print("Batch {}/{}".format(i+1, number_of_batches), end="\r")
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
    print("\nSubmission file written")


def main():
    # Enable multiple GPUs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Test
    test()


main()
