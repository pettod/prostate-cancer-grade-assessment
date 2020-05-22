import os
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np
import argparse

ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
TRAIN = os.path.join(ROOT, 'train_images')
sz = 128
N = 16
SAVE_DIR = os.path.join(ROOT, f'Iafoss-{N}-{sz}x{sz}')
parser = argparse.ArgumentParser()
parser.add_argument('--mode',  type=str, choices=['original', 'concatenated'], default='original')
args = parser.parse_args()


def tile(img):
    result = []
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
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
    return result


if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
names = [name[:-5] for name in os.listdir(TRAIN)]
for name in tqdm(names):
    img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
    tiles = tile(img)
    if(args.mode == 'concatenated'):
        concatenated_img = np.zeros((sz*4, sz*4, 3))
        for i, t in enumerate(tiles):
            img,idx = t['img'],t['idx']
            concatenated_img[128*(i//4):128*(i//4 + 1), 128*(i%4):128*(i%4 + 1)] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #if read with PIL RGB turns into BGR
        cv2.imwrite("{}.png".format(os.path.join(SAVE_DIR, name)), concatenated_img)
    else:
        for t in tiles:
            img,idx = t['img'],t['idx']
            cv2.imwrite(os.path.join(SAVE_DIR, f'{name}_{idx}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
