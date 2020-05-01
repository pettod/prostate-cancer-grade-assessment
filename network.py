from tensorflow.keras.layers import \
    Activation, Add, BatchNormalization, Conv2D, Input, Lambda, UpSampling2D, \
    Reshape, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf


def net(input_shape):
    h, w, c = input_shape
    inputs = Input(shape=input_shape)
    x = inputs
    kernels = createKernels()
    for kernel in kernels:
        x = kernel(x)
    return Model(inputs=inputs, outputs=x)


def createKernels():
    kernels = []
    kernels.append(Conv2D(32, [3, 3], padding="same", activation="relu"))
    return kernels
