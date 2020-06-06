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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class MaskGenerator(Dataset):
    def __init__(
            self, mask_directory, image_directory, train_csv_path,
            patch_size, normalize=False):
        self.__image_directory = image_directory
        self.__mask_directory = mask_directory
        self.__mask_names = []
        self.__normalize = normalize
        self.__patch_size = patch_size
        self.__train_csv_path = train_csv_path

        self.__readDatasetFileNames()

    def __cropPatchesFromImageAndMask(
            self, image_name, mask_name, downsample_level=None):
        patch_shape = (self.__patch_size, self.__patch_size)

        # downsample_level: 0, 1, 2, None (random)
        # Use only 2 or None (MultiImage is used for low resolution image,
        # OpenSlide for high resolution image (to save memory and faster
        # process, Openslide did not work for low resolution image))
        # Resolution downsample levels: 1, 4, 16
        multi_image = MultiImage(image_name)
        multi_mask = MultiImage(mask_name)
        image_slide = OpenSlide(image_name)
        mask_slide = OpenSlide(mask_name)
        if downsample_level is None:
            downsample_level = 2
            image_to_crop = multi_image[-1]
            mask_to_crop = multi_mask[-1]
        else:
            image_to_crop = multi_image[downsample_level]
            mask_to_crop = multi_mask[downsample_level]
        image_shape = tuple(image_to_crop.shape[::-1][1:])
        resolution_relation = 4 ** (2 - downsample_level)

        # Find coordinates from where to select patch
        cell_coordinates = self.__getCellCoordinatesFromImage(multi_image)

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
                image_patch = np.array(image_slide.read_region((
                    start_x, start_y), 0, patch_shape))[..., :3]
                mask_patch = np.array(mask_slide.read_region((
                    start_x, start_y), 0, patch_shape))[..., :3]
            else:
                image_patch = image_to_crop[start_y:end_y, start_x:end_x]
                mask_patch = mask_to_crop[start_y:end_y, start_x:end_x]

            # Resize if original image size was smaller than image_patch_size
            if image_patch.shape[:2] != patch_shape:
                padding = np.subtract(patch_shape, image_patch.shape[:2])
                padding = ([0, padding[0]], [0, padding[1]], [0, 0])
                image_patch = np.pad(image_patch, padding, constant_values=255)
                mask_patch = np.pad(mask_patch, padding, constant_values=0)

            # Patch has enough colored areas (not pure white)
            # Otherwise iterate again
            if np.mean(image_patch) < 230:
                break
        return image_patch, mask_patch

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

    def __readDatasetFileNames(self):
        train_csv = pd.read_csv(self.__train_csv_path)
        radboud_image_names = train_csv[
            train_csv["data_provider"] == "radboud"][
                "image_id"].values.tolist()
        radboud_image_names = [
            os.path.join(self.__mask_directory, i + "_mask.tiff") for i in
            radboud_image_names]

        existing_mask_names = list(filter(
            lambda x: os.path.exists(x), radboud_image_names))
        self.__mask_names = np.array(existing_mask_names)

    def normalizeArray(self, data_array, max_value=255):
        return ((data_array / max_value - 0.5) * 2).astype(np.float32)

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array.astype(np.uint8)

    def __len__(self):
        return len(self.__mask_names)

    def __getitem__(self, idx):
        mask_name = str(self.__mask_names[idx])
        image_name = os.path.join(
            self.__image_directory,
            mask_name.split('/')[-1].replace("_mask", ""))
        return self.__cropPatchesFromImageAndMask(image_name, mask_name)


if __name__ == "__main__":
    # Data paths
    ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
    TRAIN_X_DIR = os.path.join(ROOT, "train_images")
    TRAIN_Y_DIR = os.path.join(ROOT, "train_label_masks")
    TRAIN_CSV_PATH = os.path.join(ROOT, "train.csv")
    dataset = MaskGenerator(
        TRAIN_Y_DIR, TRAIN_X_DIR, TRAIN_CSV_PATH, patch_size=256)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    color_codes = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255)]
    for image_batch, mask_batch in dataloader:
        image = image_batch.numpy()[0]
        mask = mask_batch.numpy()[0, ..., 0]
        r = np.copy(mask)
        g = np.copy(mask)
        b = np.copy(mask)
        for i in range(len(color_codes)):
            r[r == i] = color_codes[i][0]
            g[g == i] = color_codes[i][1]
            b[b == i] = color_codes[i][2]
        mask = cv2.merge((r, g, b))
        plt.imshow(cv2.hconcat([image, mask]))
        plt.draw()
        plt.pause(2)
        plt.clf()
