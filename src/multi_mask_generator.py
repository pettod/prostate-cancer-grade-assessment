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


class MultiMaskGenerator(Dataset):
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

    def __concatenateTilePatches(self, image_patches, mask_patches):
        concatenated_image = np.ones((PATCH_SIZE*4, PATCH_SIZE*4, 3))*255
        concatenated_mask = np.zeros((PATCH_SIZE*4, PATCH_SIZE*4))
        for i in range(len(image_patches)):
            image_patch, mask_patch = image_patches[i], mask_patches[i]
            concatenated_image[128*(i//4):128*(i//4 + 1), 128*(i%4):128*(i%4 + 1)] = cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR)
            concatenated_mask[128*(i//4):128*(i//4 + 1), 128*(i%4):128*(i%4 + 1)] = mask_patch[..., 0]
        return concatenated_image, concatenated_mask

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

    def __tile(self, img_path, mask_path, number_of_tiles=12):
        img = MultiImage(img_path)[-1]
        mask = MultiImage(mask_path)[-1]
        shape = img.shape
        pad0, pad1 = (self.__patch_size - shape[0]%self.__patch_size)%self.__patch_size, (self.__patch_size - shape[1]%self.__patch_size)%self.__patch_size
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=255)
        mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=0)
        img = img.reshape(img.shape[0]//self.__patch_size,self.__patch_size,img.shape[1]//self.__patch_size,self.__patch_size,3)
        img = img.transpose(0,2,1,3,4).reshape(-1,self.__patch_size,self.__patch_size,3)
        mask = mask.reshape(mask.shape[0]//self.__patch_size,self.__patch_size,mask.shape[1]//self.__patch_size,self.__patch_size,3)
        mask = mask.transpose(0,2,1,3,4).reshape(-1,self.__patch_size,self.__patch_size,3)
        if len(img) < number_of_tiles:
            mask = np.pad(mask,[[0,number_of_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
            img = np.pad(img,[[0,number_of_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:number_of_tiles]
        img = img[idxs]
        mask = mask[idxs]
        return img, mask

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
        image_patches, mask_patches = self.__tile(image_name, mask_name)
        return self.__concatenateTilePatches(image_patches, mask_patches)


if __name__ == "__main__":
    from matplotlib import colors

    # Data paths
    ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
    TRAIN_X_DIR = os.path.join(ROOT, "train_images")
    TRAIN_Y_DIR = os.path.join(ROOT, "train_label_masks")
    TRAIN_CSV_PATH = os.path.join(ROOT, "train.csv")

    # Create Pytorch generator
    PATCH_SIZE = 128
    dataset = MultiMaskGenerator(
        TRAIN_Y_DIR, TRAIN_X_DIR, TRAIN_CSV_PATH, patch_size=PATCH_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)

    # Radboud clinic colors
    RADBOUD_COLOR_CODES = {
        "0": np.array(["0 Background",   np.array([  0,   0,   0])]),
        "1": np.array(["1 Stroma",       np.array([153, 221, 255])]),
        "2": np.array(["2 Healthy",      np.array([  0, 153,  51])]),
        "3": np.array(["3 Gleason 3",    np.array([255, 209,  26])]),
        "4": np.array(["4 Gleason 4",    np.array([255, 102,   0])]),
        "5": np.array(["5 Gleason 5",    np.array([255,   0,   0])]),
    }

    # Color bar details
    cmap = colors.ListedColormap(
        list(np.array(list(RADBOUD_COLOR_CODES.values()))[:, 1] / 255))
    grades = list(np.arange(0, 13))
    grades_descriptions = [""] * 13
    grades_descriptions[1::2] = list(np.array(list(
        RADBOUD_COLOR_CODES.values()))[:, 0])
    norm = colors.BoundaryNorm(grades, cmap.N+1)

    # Load batch
    for image_batch, mask_batch in dataloader:
        image = image_batch.numpy()[0]
        mask = mask_batch.numpy()[0]

        # Colorize mask
        r = np.copy(mask)
        g = np.copy(mask)
        b = np.copy(mask)
        for i in range(len(RADBOUD_COLOR_CODES)):
            r[r == i] = RADBOUD_COLOR_CODES[str(i)][1][0]/255
            g[g == i] = RADBOUD_COLOR_CODES[str(i)][1][1]/255
            b[b == i] = RADBOUD_COLOR_CODES[str(i)][1][2]/255
        mask = cv2.merge((r, g, b))
        image /= 255

        # Plot mask and image
        plotted_cell_mask = plt.imshow(
            cv2.hconcat([image, mask]), cmap=cmap, norm=norm)
        colorbar = plt.colorbar(plotted_cell_mask, cmap=cmap, ticks=grades)
        colorbar.ax.set_yticklabels(grades_descriptions)
        plt.draw()
        plt.pause(2)
        plt.clf()
