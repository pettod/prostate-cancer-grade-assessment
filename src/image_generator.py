import numpy as np
import math
import random
import pandas as pd
import os
import glob
import cv2
from skimage.io import MultiImage
from openslide import OpenSlide
from PIL import Image


class DataGenerator:
    def __init__(
            self, data_directory, batch_size, patch_size, patches_per_image=1,
            normalize=False, shuffle=True, rotate=False):
        self.__available_indices = []
        self.__batch_size = batch_size
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

        self.__readDatasetFileNames()

    def __concatenateTilePatches(self, batch):
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

    def __cropPatchesFromImage(self, image_name, downsample_level=None):
        patch_shape = (self.__patch_size, self.__patch_size)

        # downsample_level: 0, 1, 2, None (random)
        # Use only 2 or None (MultiImage is used for low resolution image,
        # OpenSlide for high resolution image (to save memory and faster
        # process, Openslide did not work for low resolution image))
        # Resolution downsample levels: 1, 4, 16
        multi_image = MultiImage(image_name)
        use_mixed_resolutions = False
        if downsample_level is None:
            use_mixed_resolutions = True
            image_slide = OpenSlide(image_name)
            image_to_crop = multi_image[-1]
        else:
            image_to_crop = multi_image[downsample_level]
            image_shape = tuple(image_to_crop.shape[::-1][1:])
            resolution_relation = 4 ** (2 - downsample_level)

        # Find coordinates from where to select patch
        cell_coordinates = self.__getCellCoordinatesFromImage(multi_image)

        # Crop patches
        patches = []
        for i in range(self.__patches_per_image):

            # Choose mixed down sample level (low and high (not mid))
            if use_mixed_resolutions:
                downsample_level = int(
                    i * 2 / self.__patches_per_image) * 2
                image_shape = image_slide.level_dimensions[downsample_level]
                resolution_relation = 4 ** (2 - downsample_level)

            # Iterate good patch
            for j in range(5):
                random_index = random.randint(0, cell_coordinates.shape[1] - 1)

                # Scale coordinates by the number of resolution relation
                # between low-resolution image and high/mid-resolution.
                # Take center of the cell coordinate by subtracting
                # 0.5*patch_size.
                start_y, start_x = (
                    cell_coordinates[:, random_index] * resolution_relation - 
                    int(0.5 * self.__patch_size))
                start_x = max(0, min(
                    start_x, image_shape[0] - self.__patch_size))
                start_y = max(0, min(
                    start_y, image_shape[1] - self.__patch_size))
                end_x, end_y = np.array(
                    [start_x, start_y]) + self.__patch_size

                # Crop from mid/high resolution image
                if downsample_level == 0:
                    patch = np.array(image_slide.read_region((
                        start_x, start_y), 0, patch_shape))[..., :3]
                else:
                    patch = image_to_crop[start_y:end_y, start_x:end_x]

                # Resize if original image size was smaller than patch_size
                if patch.shape[:2] != patch_shape:
                    patch = cv2.resize(
                        patch, dsize=patch_shape,
                        interpolation=cv2.INTER_LINEAR)

                # Patch has enough colored areas (not pure white)
                # Otherwise iterate again
                if np.mean(patch) < 230:
                    break
            patches.append(patch)
        return patches

    def __cropPatchesFromImages(self):
        images = []
        for i in self.__latest_used_indices:
            images.append(self.__cropPatchesFromImage(self.__image_names[i]))
        return np.moveaxis(np.array(images), 0, 1)

    def __getBatchLabels(self, categorical_labels, number_of_classes):
        # Get label integers
        if self.__data_stored_into_folders:
            y_batch = [
                int(self.__image_names[i].split('/')[-2])
                for i in self.__latest_used_indices]
        else:
            y_batch = [self.__labels[i] for i in self.__latest_used_indices]

        # Transform integers to categorical
        if categorical_labels:
            y_batch = np.array(
                [np.eye(number_of_classes)[i] for i in y_batch],
                dtype=np.float32)
        else:
            y_batch = np.array(y_batch)
        return y_batch

    def __getCellCoordinatesFromImage(self, multi_image):
        # Threshold of color value to define cell (0 to 255)
        detection_threshold = 200

        # Read low resolution image (3 images resolutions)
        low_resolution_image = multi_image[-1]
        image_shape = low_resolution_image.shape

        # Find pixels which have cell / exclude white pixels
        cell_coordinates = np.array(np.where(np.mean(
            low_resolution_image, axis=-1) < detection_threshold))

        # If image includes only white areas or very white, generate random
        # coordinates
        if cell_coordinates.shape[1] == 0:
            random_coordinates = []
            for i in range(100):
                random_x = random.randint(
                    0, image_shape[0] - self.__patch_size)
                random_y = random.randint(
                    0, image_shape[1] - self.__patch_size)
                random_coordinates.append([random_y, random_x])
            cell_coordinates = np.transpose(np.array(random_coordinates))
        return cell_coordinates

    def __pickBatchIndices(self):
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

    def __readDatasetFileNames(self):
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

    def __readSavedTilePatches(self):
        images = []
        for i in self.__latest_used_indices:
            images.append(np.array(Image.open(self.__image_names[i])))
        return np.array(images)

    def __rotateBatchImages(self, batch):
        rotated_images = []
        for i in range(batch.shape[0]):
            random_angle = random.randint(0, 3)
            rotated_images.append(np.rot90(batch[i], random_angle))
        return np.array(rotated_images)

    def getImageGeneratorAndNames(self):
        while True:
            self.__pickBatchIndices()

            # Read images
            image_names = [
                self.__image_names[i] for i in self.__latest_used_indices]
            if image_names[0].split('.')[-1] == "tiff":
                images = self.__cropPatchesFromImages()
                images = self.__concatenateTilePatches(images)
            else:
                images = self.__readSavedTilePatches()
            if self.__rotate:
                images = self.__rotateBatchImages(images)
            if self.__normalize:
                images = self.normalizeArray(images)
            yield images, image_names

    def trainImagesAndLabels(
            self, labels_file_path=None, categorical_labels=True,
            number_of_classes=6):
        if not self.__data_stored_into_folders:
            self.__labels = pd.read_csv(
                labels_file_path)["isup_grade"].values.tolist()
        batch_generator = self.getImageGeneratorAndNames()

        while True:
            X_batch, image_names = next(batch_generator)
            y_batch = self.__getBatchLabels(
                categorical_labels, number_of_classes)
            yield X_batch, y_batch

    def numberOfBatchesPerEpoch(self):
        return math.ceil(self.__image_names.shape[0] / self.__batch_size)

    def normalizeArray(self, data_array, max_value=255):
        return ((data_array / max_value - 0.5) * 2).astype(np.float32)

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array.astype(np.uint8)
