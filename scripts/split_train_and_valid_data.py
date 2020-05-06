import glob
import os
import pandas as pd


# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN_X_DIR = os.path.join(ROOT, "train_images")
TRAIN_Y_DIR = os.path.join(ROOT, "train.csv")

# Train/valid split ratio
# First X % is train, last (100-X) % is validation
TRAIN_VALID_SPLIT = 0.9


def splitCsvFiles(valid_y_dir):
    train_csv_file = pd.read_csv(TRAIN_Y_DIR)
    file_names = sorted(glob.glob(os.path.join(TRAIN_X_DIR, '*')))
    sample_split_index = int(TRAIN_VALID_SPLIT * len(file_names))
    valid_csv_file = train_csv_file[sample_split_index:]
    valid_csv_file.to_csv(valid_y_dir, index=False)
    train_csv_file = train_csv_file[:sample_split_index]
    train_csv_file.to_csv(TRAIN_Y_DIR, index=False)


def splitImages(valid_x_dir, valid_y_dir):
    valid_image_ids = pd.read_csv(valid_y_dir)["image_id"].values.tolist()
    for valid_image_id in valid_image_ids:
        image_name = valid_image_id + ".tiff"
        original_file_path = os.path.join(TRAIN_X_DIR, image_name)
        target_file_path = os.path.join(valid_x_dir, image_name)
        os.rename(original_file_path, target_file_path)
    print("{} files moved from '{}' to '{}'".format(
        len(valid_image_ids), TRAIN_X_DIR, valid_x_dir))


def isSplitDone(valid_x_dir, valid_y_dir):
    if os.path.isdir(valid_x_dir):
        raise ValueError(
            "Valid directory exists and split has probably already done")
        return True
    if os.path.isfile(valid_y_dir):
        raise ValueError(
            "Valid CSV file exists and split has probably already done")
        return True
    return False


def main():
    valid_x_dir = os.path.join(ROOT, "valid_images")
    valid_y_dir = os.path.join(ROOT, "valid.csv")
    if isSplitDone(valid_x_dir, valid_y_dir):
        return
    os.makedirs(valid_x_dir)
    splitCsvFiles(valid_y_dir)
    splitImages(valid_x_dir, valid_y_dir)


main()
