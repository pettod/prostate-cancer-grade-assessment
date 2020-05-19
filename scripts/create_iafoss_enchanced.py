import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import matplotlib.pyplot as plt
import zipfile

TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
OUT_TRAIN = 'train.zip'


class SplitAndConcatTilesMine(ImageOnlyTransform):
    def __init__(
        self, always_apply=False, p=1.0, tile_size=128, pad_value=255, tiles_in_final_image_size=(4, 4)
    ):
        super().__init__(always_apply, p)
        self.tile_size = tile_size
        self.pad_value = pad_value
        self.tiles_in_final_image_size = tiles_in_final_image_size
        self.resulted_tiles = int(self.tiles_in_final_image_size[0] * self.tiles_in_final_image_size[1])

        self.need_reverse = True if self.pad_value == 0 else False

    def pad_image(self, image):
        height, width, channels = image.shape
        pad_h, pad_w = self.tile_size - height % self.tile_size, self.tile_size - width % self.tile_size
        res_size = (
            self.tiles_in_final_image_size[0] * self.tile_size,
            self.tiles_in_final_image_size[1] * self.tile_size,
        )
        if res_size[0] > height + pad_h:
            pad_h = res_size[0] - height
        if res_size[1] > width + pad_w:
            pad_w = res_size[1] - width

        padded_img = np.pad(image, [(pad_h, 0), (pad_w, 0), (0, 0)], "constant", constant_values=self.pad_value)
        height_padded, width_padded, channels_padded = padded_img.shape

        assert height_padded >= height
        assert width_padded >= width
        assert channels_padded >= channels
        assert height_padded % self.tile_size == 0
        assert width_padded % self.tile_size == 0
        return padded_img, height_padded, width_padded

    def cut_tiles(self, padded_img, height_padded, width_padded):
        w_len, h_len = width_padded // self.tile_size, height_padded // self.tile_size
        h_w_tile_storage = [[None for w in range(w_len)] for h in range(h_len)]
        tiles = []
        for h in range(h_len):
            for w in range(w_len):
                tile = padded_img[
                    self.tile_size * h : self.tile_size * (h + 1), self.tile_size * w : self.tile_size * (w + 1)
                ]
                tile_intensivity = tile.sum()
                h_w_tile_storage[h][w] = tile
                tiles.append([tile, h, w, tile_intensivity])
        sorted_tiles = sorted(tiles, key=lambda x: x[3], reverse=self.need_reverse)
        return h_w_tile_storage, sorted_tiles

    def constract_bin_matrix(self, sorted_tiles, height_padded, width_padded):
        # fill bin_mask [intence_block_bool, height, width]
        bin_mask = np.zeros((height_padded // self.tile_size, width_padded // self.tile_size, 3), dtype=int)
        for i in range(self.resulted_tiles):
            _, h, w, _ = sorted_tiles[i]
            bin_mask[h][w][0] = 1
            bin_mask[h][w][1] = h
            bin_mask[h][w][2] = w
        return bin_mask

    def apply(self, image, **params):

        padded_img, height_padded, width_padded = self.pad_image(image)

        h_w_tile_storage, sorted_tiles = self.cut_tiles(padded_img, height_padded, width_padded)

        bin_mask = self.constract_bin_matrix(sorted_tiles, height_padded, width_padded)

        resulted_img = [
            [None for _ in range(self.tiles_in_final_image_size[1])] for _ in range(self.tiles_in_final_image_size[0])
        ]
        region_of_interest = np.ones(self.tiles_in_final_image_size, dtype=bool)

        most_intencivity = 1 # crunch for while loop
        while most_intencivity > 0:
            bin_mask, region_of_interest, resulted_img, most_intencivity = self.process_region(
                bin_mask, region_of_interest, resulted_img
            )

        # deal with leftovers
        bin_mask, resulted_img
        bin_h, bin_w, _ = bin_mask.shape
        for h in range(bin_h):
            for w in range(bin_w):
                if bin_mask[h][w][0] == 1:
                    resulted_img = self.insert_value_in_res_im_array(resulted_img, bin_mask[h][w][1:].tolist())
                    bin_mask[h][w][0] = 0

        tiles_arr = [
            [None for _ in range(self.tiles_in_final_image_size[1])] for _ in range(self.tiles_in_final_image_size[0])
        ]
        for h in range(self.tiles_in_final_image_size[0]):
            for w in range(self.tiles_in_final_image_size[1]):
                target_h, target_w = resulted_img[h][w]
                tiles_arr[h][w] = h_w_tile_storage[target_h][target_w]

        return np.hstack(np.hstack(np.array(tiles_arr)))

    def get_transform_init_args_names(self):
        return ("tile_size", "pad_value", "tiles_in_final_image_size",)

    def insert_value_in_res_im_array(self, resulted_img, value):
        for h in range(len(resulted_img)):
            for w in range(len(resulted_img[0])):
                if resulted_img[h][w] is None:
                    resulted_img[h][w] = value
                    return resulted_img

    def process_region(self, bin_mask, region_of_interest, resulted_img):
        # select_region
        most_intensivity = 0
        most_intensive_region = None

        bin_mask_h, bin_mask_w, _ = bin_mask.shape
        for h in range(bin_mask_h - self.tiles_in_final_image_size[0]):
            for w in range(bin_mask_w - self.tiles_in_final_image_size[1]):
                h_slice = slice(h, h + self.tiles_in_final_image_size[0])
                w_slice = slice(w, w + self.tiles_in_final_image_size[1])
                bin_tile = bin_mask[h_slice, w_slice]
                intense = bin_tile[region_of_interest, 0].sum()
                if intense > most_intensivity:
                    most_intensivity = intense
                    most_intensive_region = bin_tile
                    most_intensive_region_slices = (h_slice, w_slice)
        if most_intensivity > 0:
            # fill resulted arr
            new_region_of_interest = np.zeros(self.tiles_in_final_image_size, dtype=bool)
            for h in range(self.tiles_in_final_image_size[0]):
                for w in range(self.tiles_in_final_image_size[1]):
                    interest = region_of_interest[h, w]
                    if interest:
                        is_filled_tile = most_intensive_region[h][w][0]
                        if is_filled_tile:
                            resulted_img[h][w] = most_intensive_region[h][w][1:].tolist()  # tolist important
                        else:
                            new_region_of_interest[h][w] = 1

            # clean selected
            bin_mask[most_intensive_region_slices][region_of_interest] = 0

            return bin_mask, new_region_of_interest, resulted_img, most_intensivity
        else:
            return bin_mask, region_of_interest, resulted_img, most_intensivity

def create_enchanced_iafoss():
    names = [name[:-5] for name in os.listdir(TRAIN)]
    imgs = []
    for name in tqdm(names):
        img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
        imgs.append(img)

    mine_tiler = SplitAndConcatTilesMine()
    with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
        for i, name in enumerate(names):
            img = mine_tiler(image=imgs[i])["image"]
            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}.png', img)

        img_out.extractall('../input/prostate-cancer-grade-assessment/iafoss_enchanced')

create_enchanced_iafoss()