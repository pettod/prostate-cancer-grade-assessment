from tensorflow.keras.layers import \
    Activation, Add, BatchNormalization, Conv2D, Input, Lambda, UpSampling2D, \
    Reshape, concatenate, Conv2DTranspose, Dense, Flatten, \
    GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
import math


def net(input_shape, patches_per_image=1):
    patch = Input(input_shape)
    x = Conv2D(32, 5, activation="relu")(patch)
    x = resnetBlock(x, 32, 5)
    x = resnetBlock(x, 32, 5)
    x = resnetBlock(x, 32, 5)
    x = Conv2D(64, 5, strides=2, activation="relu")(x)
    x = resnetBlock(x, 64, 5)
    x = resnetBlock(x, 64, 5)
    x = resnetBlock(x, 64, 5)
    x = Conv2D(128, 5, strides=2, activation="relu")(x)
    x = resnetBlock(x, 128, 5)
    x = resnetBlock(x, 128, 5)
    x = resnetBlock(x, 128, 5)
    x = Conv2D(256, 5, strides=2, activation="relu")(x)
    x = resnetBlock(x, 256, 5)
    x = resnetBlock(x, 256, 5)
    x = resnetBlock(x, 256, 5)
    x = Flatten()(x)
    x = Dense(6, activation="softmax", use_bias=False)(x)
    return Model(patch, x)


def resnetBlock(x, dimensions, kernel_size):
    resnet_1 = Conv2D(
        dimensions, kernel_size, activation="relu", padding="same")(x)
    resnet_2 = Conv2D(
        dimensions, kernel_size, activation=None, padding="same")(resnet_1)
    return Add()([x, resnet_2])


def concatenateSquare(patches):
    patches_per_side = int(math.sqrt(len(patches)))
    concatenated_patches = []
    for i in range(patches_per_side):
        concatenated_patches.append(
            Concatenate(axis=1)(patches[:patches_per_side]))
        patches = patches[patches_per_side:]
    return Concatenate(axis=2)(concatenated_patches)
