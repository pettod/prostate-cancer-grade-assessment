import openslide
import cv2
import os
import csv
import numpy as np

ROOT = os.path.realpath('../input/prostate-cancer-grade-assessment/')
TRAIN_IMAGES = os.path.join(ROOT, 'train_images')
TRAIN_LABEL_SEGMENTAION_MASKS = os.path.join(ROOT, 'train_label_masks')

labels = []

RADBOUD_COLOR_CODES = [
    (  0,   0,   0),  # Nothing
    (153, 221, 255),  # Stroma
    (  0, 153,  51),  # Healthy
    (255, 209,  26),  # Gleason 3
    (255, 102,   0),  # Gleason 4
    (255,   0,   0)   # Gleason 5
]

KAROLINSKA_COLOR_CODES = [
    (  0,   0,   0),  # Nothing
    (  0, 255,   0),  # Healthy
    (255,   0,   0),  # Cancer
]


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

    return img, colorCodeMask(mask, idx)


def getIsupGradeByIndex(idx):
    return labels[idx+1][2]


def getImgNameByIndex(idx):
    return labels[idx+1][0]


def getClinicByIndex(idx):
    return labels[idx+1][1]


def zoom(event, x, y, flags, data):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    idx, width = data
    RANGE = 256

    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_LABEL_SEGMENTAION_MASKS, getImgNameByIndex(idx) + '_mask.tiff'))

    mask = np.array(tiff.read_region(
        (0, 0), 2, tiff.level_dimensions[2]))[:, :, 0]

    if x >= width:
        x -= width

    y_min, _ = min_max_mask_coordinates(mask, axis=1)
    x_min, _ = min_max_mask_coordinates(mask, axis=0)

    mask = np.array(tiff.read_region(
        (16*(x + x_min) - RANGE, 16*(y + y_min) - RANGE),
        0, (RANGE*2, RANGE*2)))[:, :, 0]

    tiff = openslide.OpenSlide(os.path.join(
        TRAIN_IMAGES, getImgNameByIndex(idx) + '.tiff'))
    img = np.array(tiff.read_region(
        (16*(x + x_min) - RANGE, 16*(y + y_min) - RANGE),
        0, (RANGE*2, RANGE*2)))

    mask = colorCodeMask(mask, idx)

    alpha = 0.65
    transparent_image = np.copy(mask)
    while True:
        bgr_image = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        cv2.addWeighted(
            bgr_image, alpha, mask, 1 - alpha, 0, transparent_image)
        concat_image = cv2.hconcat([bgr_image, mask, transparent_image])
        cv2.imshow("Prostate_zoom", concat_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyWindow("Prostate_zoom")
            break


def colorCodeMask(mask, idx):
    r = np.copy(mask)
    g = np.copy(mask)
    b = np.copy(mask)
    if getClinicByIndex(idx) == 'radboud':
        for i in range(len(RADBOUD_COLOR_CODES)):
            r[r == i] = RADBOUD_COLOR_CODES[i][0]
            g[g == i] = RADBOUD_COLOR_CODES[i][1]
            b[b == i] = RADBOUD_COLOR_CODES[i][2]
    else:
        for i in range(len(KAROLINSKA_COLOR_CODES)):
            r[r == i] = KAROLINSKA_COLOR_CODES[i][0]
            g[g == i] = KAROLINSKA_COLOR_CODES[i][1]
            b[b == i] = KAROLINSKA_COLOR_CODES[i][2]
    return cv2.merge((b, g, r))


def main():
    readCSV()

    idx = 0 #int(input("Input the image index: "))
    while idx != -1:
        while getClinicByIndex(idx) != "radboud":
            idx += 1
        print("Index:      %s" % idx)
        print("ISUP:       %s" % getIsupGradeByIndex(idx))
        print("Image name: %s" % getImgNameByIndex(idx))
        print("Clinic:     %s" % getClinicByIndex(idx))
        print()
        img, mask = getProstateByIndex(idx)
        img, mask = trim_image_to_mask_size(img, mask)
        while True:
            bgr_image = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
            concat_image = cv2.hconcat([bgr_image, mask])
            cv2.imshow("Prostate", concat_image)
            cv2.setMouseCallback("Prostate", zoom, (idx, img.shape[1]))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        idx += 1


main()
