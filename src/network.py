from tensorflow.keras.layers import \
    Activation, Add, BatchNormalization, Conv2D, Input, Lambda, UpSampling2D, \
    Reshape, concatenate, Conv2DTranspose, Dense, Flatten, \
    GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2


def net(input_shape, patches_per_image):
    patches = [Input(input_shape) for i in range(patches_per_image)]
    branches = [branch(patch) for patch in patches]
    merge = Concatenate(axis=-1)(branches)
    dense = Dense(256)(merge)
    dropout = Dropout(0.3)(dense)
    output = Dense(6, activation="softmax")(dropout)
    return Model(patches, output)


def branch(input_image):
    x = Conv2D(128, (3, 3))(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    return Dropout(0.3)(x)
