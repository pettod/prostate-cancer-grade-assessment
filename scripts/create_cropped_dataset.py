import cv2
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import time

from src.image_generator import DataGenerator


# Paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_images")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
PROGRAM_TIME_STAMP = time.strftime("%Y-%m-%d_%H%M%S")
SAVE_DIR = os.path.join(ROOT, PROGRAM_TIME_STAMP)

# Batch details
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
CONCATENATE_PATCHES = True
SPLIT_IMAGES_INTO_CLASS_FOLDERS = True

NUMBER_OF_SAMPLES_PER_IMAGE = 10


def getImageClass(csv_file, image_id, id_index):
    row = csv_file.loc[id_index, :]
    if row["image_id"] != image_id:
        raise ValueError("Image IDs don't match: \n{} != {}".format(
            row["image_id"], image_id))
    return row["isup_grade"]


def saveImage(csv_file, image_names, batch, id_index, sample_index):
    patch = batch[0]
    image_id = image_names[0].split('/')[-1].split('.')[0]
    image_class = getImageClass(csv_file, image_id, id_index)
    image = Image.fromarray(patch)
    if NUMBER_OF_SAMPLES_PER_IMAGE > 1:
        image_id += "_{}".format(sample_index)
    image_name = image_id + ".png"
    if SPLIT_IMAGES_INTO_CLASS_FOLDERS:
        image_path = os.path.join(SAVE_DIR, *[str(image_class), image_name])
    else:
        image_path = os.path.join(SAVE_DIR, image_name)
    image.save(image_path)


def getETA(start_time, number_of_images, id_index, sample_index):
    # Compute estimated time of arrival
    time_spent = (time.time() - start_time) / 60
    number_of_computed_images = sample_index * number_of_images + id_index + 1
    number_of_left_images = \
        number_of_images * NUMBER_OF_SAMPLES_PER_IMAGE - \
        number_of_computed_images
    time_per_image = time_spent / number_of_computed_images
    estimated_time_of_arrival = time_per_image * number_of_left_images
    return estimated_time_of_arrival, time_spent


def main():
    # Cannot allow to save images into one folder if taking multiple samples
    # per image
    if NUMBER_OF_SAMPLES_PER_IMAGE > 1 and not SPLIT_IMAGES_INTO_CLASS_FOLDERS:
        raise ValueError((
            "NUMBER_OF_SAMPLES_PER_IMAGE={} must be 1 if " +
            "SPLIT_IMAGES_INTO_CLASS_FOLDERS=False").format(
                NUMBER_OF_SAMPLES_PER_IMAGE))

    # Load data generator
    test_generator = DataGenerator(
        TRAIN_X_DIR, 1, PATCH_SIZE, PATCHES_PER_IMAGE,
        concatenate_patches=CONCATENATE_PATCHES)
    test_batch_generator = test_generator.getImageGeneratorAndNames(
        normalize=False, shuffle=False)
    number_of_images = test_generator.numberOfBatchesPerEpoch()

    # Create new folders
    os.makedirs(SAVE_DIR)
    if SPLIT_IMAGES_INTO_CLASS_FOLDERS:
        for i in range(6):
            os.makedirs(os.path.join(SAVE_DIR, str(i)))
    csv_file = pd.read_csv(TRAIN_Y_DIR)

    # Save batch images N times
    start_time = time.time()
    for sample_index in range(NUMBER_OF_SAMPLES_PER_IMAGE):
        for id_index in range(number_of_images):
            batch, image_names = next(test_batch_generator)
            saveImage(csv_file, image_names, batch, id_index, sample_index)
            eta, time_spent = getETA(
                start_time, number_of_images, id_index, sample_index)
            print((
                " Round {}/{}. Image: {}/{}. Time spent: {:.1f}min. ETA: " +
                "{:.1f}min").format(
                sample_index+1, NUMBER_OF_SAMPLES_PER_IMAGE, id_index+1,
                number_of_images, time_spent, eta), end="\r")
    print()


main()
