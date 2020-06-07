import cv2
from tqdm import tqdm
import fastai
from fastai.vision import *
import os
from mish_activation import *
import warnings
# warnings.filterwarnings("ignore")
import skimage.io
import numpy as np
import pandas as pd
#from src.image_generator import DataGenerator
#from src.pytorch.network import Net
sys.path.insert(0, '../input/semisupervised-imagenet-models/semi-supervised-ImageNet1K-models-master/')
from hubconf import *

DATA = '../input/prostate-cancer-grade-assessment/train_images'
TEST = '../input/prostate-cancer-grade-assessment/test.csv'
SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
MODELS = ['models/pytorch/2020-05-27_171106/model.pt'] #[f'../input/pretrained_models/RNXT50_{i}.pth' for i in range(4)]

sz = 128
bs = 1
N = 12
nworkers = 2
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
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(
            self, data_directory, batch_size, patch_size, patches_per_image=1,
            normalize=False, drop_last_batch=False, shuffle=True, rotate=False,
            add_gaussian_blur=False):
        self.__add_gaussian_blur = add_gaussian_blur
        self.__available_indices = []
        self.__batch_size = batch_size
        self.__data_directory = data_directory
        self.__data_stored_into_folders = None
        self.__drop_last_batch = drop_last_batch
        self.__image_names = []
        self.__labels = None
        self.__latest_used_indices = []
        self.__normalize = normalize
        self.__patch_size = patch_size
        self.__patches_per_image = patches_per_image
        self.__rotate = rotate
        self.__shuffle = shuffle

        # Update this if statement if adding more augmentations
        self.__add_augmentations = False
        if self.__rotate or self.__add_gaussian_blur:
            self.__add_augmentations = True

        self.__readDatasetFileNames()

    def __addAugmentationForBatchImages(self, batch):
        augmented_images = []
        for i in range(batch.shape[0]):
            image = batch[i]
            if self.__rotate:
                random_angle = random.randint(0, 3)
                image = np.rot90(image, random_angle)
            if self.__add_gaussian_blur and random.randint(0, 9):
                kernel_size = random.randrange(3, 11, 2)
                image = cv2.GaussianBlur(
                    image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
            augmented_images.append(image)
        return np.array(augmented_images)

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
                    padding = np.subtract(patch_shape, patch.shape[:2])
                    padding = ([0, padding[0]], [0, padding[1]], [0, 0])
                    patch = np.pad(patch, padding, constant_values=255)

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
        # Initialize all indices to be available
        if len(self.__available_indices) == 0 or (
                self.__drop_last_batch and
                len(self.__available_indices) < self.__batch_size):
            self.__available_indices = list(
                np.arange(0, self.__image_names.shape[0]))

        # Take batch
        if self.__batch_size < len(self.__available_indices):

            # Random images in batch
            if self.__shuffle:
                random_indices_from_list = random.sample(
                    range(len(self.__available_indices)), self.__batch_size)
                self.__latest_used_indices = []
                for i in random_indices_from_list:
                    self.__latest_used_indices.append(
                        self.__available_indices[i])

            # Batch images in alphabetical order
            else:
                self.__latest_used_indices = self.__available_indices[
                    :self.__batch_size].copy()

        # Take last batch
        else:
            self.__latest_used_indices = self.__available_indices.copy()

        # Remove used indices from availabe indices
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
            if self.__add_augmentations:
                images = self.__addAugmentationForBatchImages(images)
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
        if self.__drop_last_batch:
            return int(self.__image_names.shape[0] / self.__batch_size)
        else:
            return math.ceil(self.__image_names.shape[0] / self.__batch_size)

    def normalizeArray(self, data_array, max_value=255):
        return ((data_array / max_value - 0.5) * 2).astype(np.float32)

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array.astype(np.uint8)

    def __len__(self):
        return self.__image_names.shape[0]

    def __getitem__(self, idx):
        self.__latest_used_indices = [idx]
        name = self.__image_names[idx]
        # print(os.path.join(DATA,name+'.tiff'))
        image = self.__cropPatchesFromImages()
        image = self.__concatenateTilePatches(image)
        image = self.normalizeArray(image)

        return image, name

def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # model.load_state_dict(torch.load(model_name))
    # state_dict = #load_state_dict_from_url(url, progress=progress)
    # model.load_state_dict(state_dict)
    return model

class Net(nn.Module):
    def __init__(self, arch='resnext50_32x4d', n=6, pre=True):
        super().__init__()
        m = _resnext(semi_supervised_model_urls[arch], Bottleneck, [3, 4, 6, 3], False, 
                progress=False,groups=32,width_per_group=4) 
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
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # x: bs*N x C x 4 x 4
        shape = x.shape
        # Concatenate the output for tiles into a single map
        x = x.view(-1, n, shape[1], shape[2], shape[3]).permute(
            0, 2, 1, 3, 4).contiguous().view(
                -1, shape[1], shape[2]*n, shape[3])
        # x: bs x C x N*4 x 4
        x = self.adaptive_concat_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        # x = self.batchnorm(x)
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

models = []
for path in MODELS:
    model = nn.DataParallel(Net()).to('cpu')
    model.load_state_dict(torch.load(path))
    model.float()
    model.eval()
    model.cuda()
    models.append(model)

def tile(img):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img

mean = torch.tensor([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304])
std = torch.tensor([0.36357649, 0.49984502, 0.40477625])

class PandaDataset(Dataset):
    def __init__(self, path, test):
        self.path = path
        self.names = list(pd.read_csv(test).image_id)
        print(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        print(os.path.join(DATA,name+'.tiff'))
        img = skimage.io.MultiImage(os.path.join(DATA,name+'.tiff'))[-1]

        tiles = torch.Tensor(1.0 - tile(img)/255.0)
        tiles = (tiles - mean)/std
        return tiles.permute(0,3,1,2), name

sub_df = pd.read_csv(SAMPLE)
if os.path.exists(DATA):
    ds = DataGenerator(
        DATA, batch_size=bs, patch_size=128, patches_per_image=16,
        normalize=True, drop_last_batch=False, shuffle=False, rotate=False,
        add_gaussian_blur=False
    )   
    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
    names,preds = [],[]

    with torch.no_grad():
        for x,y in tqdm(dl):
            x = x.cuda()
           
            x = x.squeeze(0)
            x = x.transpose(1, 3)
            p = models[0](x)

            names.append(y)
            preds.append(p)
    
    names = np.concatenate(names)
    preds = torch.cat(preds).numpy()
    sub_df = pd.DataFrame({'image_id': names, 'isup_grade': preds})
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()

sub_df.to_csv("submission.csv", index=False)
sub_df.head()