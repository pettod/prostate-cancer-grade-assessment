import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import math
import random
import numpy as np
import os
import pandas as pd
import cv2
from PIL import Image
from openslide import OpenSlide

# Data paths
ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TEST_X_DIR = os.path.join(ROOT, "test_images")

# Model parameters
MODEL_PATH = None
BATCH_SIZE = 32
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 16
CONCATENATE_PATCHES = True


class Net(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load(
            'facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.adaptive_concat_pool = AdaptiveConcatPool2d()
        self.linear_1 = nn.Sequential(nn.Linear(2*nc, 512), Mish())
        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(512, n)

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x, 1).view(-1, shape[1], shape[2], shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.adaptive_concat_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *(torch.tanh(F.softplus(x)))


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`." # from pytorch
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


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

    def __getBatchLabels(self, number_of_classes):
        # Get label integers
        if self.__data_stored_into_folders:
            y_batch = [
                int(self.__image_names[i].split('/')[-2])
                for i in self.__latest_used_indices]
        else:
            y_batch = [self.__labels[i] for i in self.__latest_used_indices]

        # Transform integers to categorical
        return np.array(y_batch)

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
                if self.__concatenate_patches:
                    images = self.__concatenateTilePatches(images)
            else:
                images = self.__readSavedTilePatches()
            if self.__rotate:
                images = self.__rotateBatchImages(images)
            if self.__normalize:
                images = self.normalizeArray(images)
            yield images, image_names

    def trainImagesAndLabels(
            self, labels_file_path=None, number_of_classes=6):
        if not self.__data_stored_into_folders:
            self.__labels = pd.read_csv(
                labels_file_path)["isup_grade"].values.tolist()
        batch_generator = self.getImageGeneratorAndNames()

        while True:
            X_batch, image_names = next(batch_generator)
            y_batch = self.__getBatchLabels(number_of_classes)
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


class Test():
    def __init__(self, device):
        self.device = device
        self.model_root = "models"
        self.model = self.loadModel()

        # Define test batch generator
        test_generator = DataGenerator(
            TEST_X_DIR, BATCH_SIZE, PATCH_SIZE, PATCHES_PER_IMAGE,
            concatenate_patches=CONCATENATE_PATCHES, normalize=True,
            rotate=False, shuffle=False)
        self.test_batch_generator = test_generator.getImageGeneratorAndNames()
        self.number_of_test_batches = test_generator.numberOfBatchesPerEpoch()

    def loadModel(self):
        # Load latest model
        if MODEL_PATH is None:
            model_name = sorted(glob.glob(os.path.join(
                self.model_root, *['*', "*.pt"])))[-1]
        else:
            if type(MODEL_PATH) == int:
                model_name = sorted(glob.glob(os.path.join(
                    self.model_root, *['*', "*.pt"])))[MODEL_PATH]
            else:
                model_name = MODEL_PATH
        model = nn.DataParallel(Net()).to(self.device)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        print("Loaded model: {}".format(model_name))
        print("{:,} model parameters".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model

    def test(self):
        submission_file = pd.read_csv(os.path.join(
            ROOT, "sample_submission.csv"))
        predictions = []
        image_names = []
        with torch.no_grad():
            for i in range(self.number_of_test_batches):
                print("Batch {}/{}".format(
                    i+1, self.number_of_test_batches), end="\r")
                batch, batch_image_names = next(self.test_batch_generator)
                batch = torch.from_numpy(np.moveaxis(
                    batch, -1, 1)).to(self.device)
                predictions += list(np.argmax(self.model(batch).cpu(), axis=1))
                image_names += batch_image_names

        # Write submission file
        for i in range(len(predictions)):
            submission_file.at[i, "image_id"] = \
                image_names[i].split('/')[-1].split('.')[0]
            submission_file.at[i, "isup_grade"] = predictions[i]
        submission_file.to_csv("submission.csv", index=False)
        submission_file.head()
        print("\nSubmission file written")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Running on CPU\n\n\n\n")

    train = Test(device)
    train.test()


main()
