import os
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np
import argparse
import pandas as pd

ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN = os.path.join(ROOT, 'train_images')
TRAIN_CSV_PATH = os.path.join(ROOT, "train.csv")
MASKS = os.path.join(ROOT, "train_label_masks/")
sz = 128
parser = argparse.ArgumentParser()

parser.add_argument('--mode',  type=str, choices=['original', 'concatenated'], default='original')
parser.add_argument('--N',  type=int, default='12')
parser.add_argument('--row_size',  type=int, default='4')


args = parser.parse_args()

N = args.N
row_size = args.row_size

IMAGE_SAVE_DIR = os.path.join(ROOT, f'Iafoss-{N}-{sz}x{sz}-{args.mode}')
MASK_SAVE_DIR = os.path.join(ROOT, f'Iafoss-{N}-{sz}x{sz}-{args.mode}-masks')

def tile(img, mask):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


if not os.path.isdir(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
if not os.path.isdir(MASK_SAVE_DIR):
    os.makedirs(MASK_SAVE_DIR)
names = [name.replace(".tiff", "") for name in os.listdir(TRAIN)]
train_csv = pd.read_csv(TRAIN_CSV_PATH)
for name in tqdm(names):
    mask_name = os.path.join(MASKS,name+'_mask.tiff')
    if not os.path.exists(mask_name):
        continue
    if not train_csv.loc[train_csv["image_id"] == name]["data_provider"].values == "radboud":
        continue
    img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
    mask = skimage.io.MultiImage(mask_name)[-1]
    tiles = tile(img, mask)
    if(args.mode == 'concatenated'):
        concatenated_img = np.zeros((sz*4, sz*4, 3))
        concatenated_mask = np.zeros((sz*4, sz*4))
        for i, t in enumerate(tiles):
            img,mask,idx = t['img'],t['mask'],t['idx']
            concatenated_img[128*(i//4):128*(i//4 + 1), 128*(i%4):128*(i%4 + 1)] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            concatenated_mask[128*(i//4):128*(i//4 + 1), 128*(i%4):128*(i%4 + 1)] = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("{}.png".format(os.path.join(IMAGE_SAVE_DIR, name)), concatenated_img)
        cv2.imwrite("{}.png".format(os.path.join(MASK_SAVE_DIR, name)), concatenated_mask)
    else:
        for t in tiles:
            img,mask,idx = t['img'],t['mask'],t['idx']
            cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, f'{name}_{idx}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(MASK_SAVE_DIR, f'{name}_{idx}.png'), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
