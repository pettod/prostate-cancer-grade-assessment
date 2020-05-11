import openslide
import cv2
import os
import csv
import numpy as np

ROOT = os.path.realpath('../input/prostate-cancer-grade-assessment/')
TRAIN_IMAGES = os.path.join(ROOT, 'train_images')
TRAIN_LABEL_SEGMENTAION_MASKS = os.path.join(ROOT, 'train_label_masks')

labels = []


def readCSV():
    global labels
    with open(os.path.join(ROOT, 'train.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(row)
    labels = sorted(labels)


def min_max_mask_coordinates(mask, axis=1):
    xy = mask.sum(axis=axis)
    xy = np.nonzero(xy)
    xy_min = np.min(xy)
    xy_max = np.max(xy)
    return xy_min, xy_max


def trim_image_to_mask_size(image, mask):
    x_min, x_max = min_max_mask_coordinates(mask, axis=1)
    y_min, y_max = min_max_mask_coordinates(mask, axis=0)

    image = image[x_min:x_max, y_min:y_max]
    mask = mask[x_min:x_max, y_min:y_max]
    return image, mask


def getProstateByIndex(idx):
    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_IMAGES, getImgNameByIndex(idx) + '.tiff'))
    img = np.array(tiff.read_region((0, 0), 2, tiff.level_dimensions[2]))

    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_LABEL_SEGMENTAION_MASKS, getImgNameByIndex(idx) + '_mask.tiff'))
    mask = np.array(tiff.read_region(
        (0, 0), 2, tiff.level_dimensions[2]))[:, :, 0]

    img, mask = trim_image_to_mask_size(img, mask)

    if getClinicByIndex(idx) == 'radboud':
        g = (mask == 1).astype(np.uint8) * 50 + \
            (mask == 2).astype(np.uint8) * 255
        b = (mask == 3).astype(np.uint8) * 255
        r = (mask > 3).astype(np.uint8) * 255
        mask = cv2.merge((b, g, r))
    else:
        b = (mask == 1).astype(np.uint8) * 255
        r = (mask == 2).astype(np.uint8) * 255
        g = mask * 0
        mask = cv2.merge((b, g, r))
    return img, mask


def getIsupGradeByIndex(idx):
    return labels[idx+1][2]


def getImgNameByIndex(idx):
    return labels[idx+1][0]


def getClinicByIndex(idx):
    return labels[idx+1][1]


def zoom(event, x, y, flags, idx):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    RANGE = 256
    
    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_LABEL_SEGMENTAION_MASKS, getImgNameByIndex(idx) + '_mask.tiff'))
    
    mask = np.array(tiff.read_region(
        (0, 0), 2, tiff.level_dimensions[2]))[:, :, 0]

    # X AND Y ARE SWAPPED BECAUSE THE 1ST AXIS IS Y, NOT X 
    y_min, _ = min_max_mask_coordinates(mask, axis=1)
    x_min, _ = min_max_mask_coordinates(mask, axis=0)
    
    
    mask = np.array(tiff.read_region(
        (16*(x + x_min) - RANGE, 16*(y + y_min) - RANGE), 0, (RANGE*2, RANGE*2)))[:, :, 0]

    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_IMAGES, getImgNameByIndex(idx) + '.tiff'))
    img = np.array(tiff.read_region((16*(x + x_min) - RANGE, 16*(y + y_min) - RANGE), 0, (RANGE*2, RANGE*2)))

    if getClinicByIndex(idx) == 'radboud':
        g = (mask == 1).astype(np.uint8) * 50 + \
            (mask == 2).astype(np.uint8) * 255
        b = (mask == 3).astype(np.uint8) * 255
        r = (mask > 3).astype(np.uint8) * 255
        mask = cv2.merge((b, g, r))
    else:
        b = (mask == 1).astype(np.uint8) * 255
        r = (mask == 2).astype(np.uint8) * 255
        g = mask * 0
        mask = cv2.merge((b, g, r))

    while True:
        cv2.imshow("Mask_zoom", mask)
        cv2.imshow("Prostate_zoom", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyWindow("Mask_zoom")
            cv2.destroyWindow("Prostate_zoom")
            break    


readCSV()

idx = int(input())
while idx != -1:
    print("Index:      %s" % idx)
    print("ISUP:       %s" % getIsupGradeByIndex(idx))
    print("Image name: %s" % getImgNameByIndex(idx))
    print("Clinic:     %s" % getClinicByIndex(idx))
    print()
    img, mask = getProstateByIndex(idx)
    img, mask = trim_image_to_mask_size(img, mask)
    while True:
        cv2.imshow("Mask", mask)
        cv2.imshow("Prostate", img)
        cv2.setMouseCallback("Prostate", zoom, idx)
        cv2.setMouseCallback("Mask", zoom, idx)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    # idx = int(input())
    idx += 1
