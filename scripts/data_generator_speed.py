import os
import time
from src.image_generator import DataGenerator


ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_images/")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")
BATCH_SIZE = 8
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16


def main():
    train_generator = DataGenerator(
        TRAIN_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE)
    train_batch_generator = train_generator.trainImagesAndLabels(
        TRAIN_Y_DIR, normalize=True)

    number_of_generated_batches = 20
    start_time = time.time()
    for i in range(number_of_generated_batches):
        batch = next(train_batch_generator)
        time_per_batch = (time.time() - start_time) / (i+1)
        print(" Batch {}/{}. Load time per batch {:.3f}s".format(
            i+1, number_of_generated_batches, time_per_batch), end="\r")
    print()


main()
