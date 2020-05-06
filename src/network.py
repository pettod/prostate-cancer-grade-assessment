from tensorflow.keras.layers import \
    Activation, Add, BatchNormalization, Conv2D, Input, Lambda, UpSampling2D, \
    Reshape, concatenate, Conv2DTranspose, Dense, Flatten, \
    GlobalAveragePooling2D, Dropout, Concatenate, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
import math


def get_model(
        base_model, layer, input_shape, classes=6,
        activation="softmax", dropout=None, pooling="avg", weights=None,
        pretrained="imagenet"):
    base = base_model(input_shape=input_shape,
                      include_top=False,
                      weights=pretrained)
    if pooling == "avg":
        x = GlobalAveragePooling2D()(base.output)
    elif pooling == "max":
        x = GlobalMaxPooling2D()(base.output)
    elif pooling is None:
        x = Flatten()(base.output)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(classes, activation=activation)(x)
    model = Model(inputs=base.input, outputs=x)
    if weights is not None:
        model.load_weights(weights)
    for l in model.layers[:layer]:
        l.trainable = False
    return model


def net(input_shape, patches_per_image=1):
    return get_model(InceptionResNetV2, 0, input_shape, dropout=None)
