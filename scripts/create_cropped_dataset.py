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
SPLIT_IMAGES_INTO_CLASS_FOLDERS = False


def getImageClass(csv_file, image_name, i):
    row = csv_file.loc[i, :]
    if row["image_id"] != image_name:
        raise ValueError("Image IDs don't match: \n{} != {}".format(
            row["image_id"], image_name))
    return row["isup_grade"]


def main():
    test_generator = DataGenerator(
        TRAIN_X_DIR, 1, PATCH_SIZE, PATCHES_PER_IMAGE,
        concatenate_patches=CONCATENATE_PATCHES)
    test_batch_generator = test_generator.getImageGeneratorAndNames(
        normalize=False, shuffle=False)
    number_of_images = test_generator.numberOfBatchesPerEpoch()
    os.makedirs(SAVE_DIR)
    if SPLIT_IMAGES_INTO_CLASS_FOLDERS:
        for i in range(6):
            os.makedirs(os.path.join(SAVE_DIR, str(i)))
    csv_file = pd.read_csv(TRAIN_Y_DIR)

    start_time = time.time()
    for i in range(number_of_images):
        batch, image_names = next(test_batch_generator)
        patch = batch[0]
        image_name = image_names[0].split('/')[-1].split('.')[0]
        image_class = getImageClass(csv_file, image_name, i)
        image = Image.fromarray(patch)
        if SPLIT_IMAGES_INTO_CLASS_FOLDERS:
            image_path = os.path.join(
                SAVE_DIR, *[str(image_class), image_name + ".png"])
        else:
            image_path = os.path.join(
                SAVE_DIR, image_name + ".png")
        image.save(image_path)
        time_spent = (time.time() - start_time) / 60
        estimated_time_of_arrival = \
            time_spent * (number_of_images - (i+1)) / (i+1)
        print("Image: {}/{}. Time spent: {:.1f}min. ETA: {:.1f}min".format(
            i+1, number_of_images, time_spent, estimated_time_of_arrival),
            end="\r")
    print()


main()
