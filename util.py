"""
Author: Sigve Rokenes
Date: February, 2019

Utility functions for variational autoencoder

"""

import skimage as sk
from skimage import io
import tensorflow as tf
import numpy as np


# ===================================== #
#                                       #
#           Utility Functions           #
#                                       #
# ===================================== #

def print_tensor(tensor, name):
    shape = tensor.get_shape()
    size = 1
    for dim in shape.as_list():
        if dim is not None:
            size *= int(dim)
    print("{:8} {:20} {:8}".format(name, str(shape), str(size)))


def conv(source, filters, strides=2, kernel=3, activation=tf.nn.leaky_relu):
    return tf.layers.conv2d(source, filters, kernel,
                            strides=strides, padding="same", activation=activation)


def deconv(source, filters, strides=2, kernel=3, activation=tf.nn.leaky_relu):
    return tf.layers.conv2d_transpose(source, filters, kernel,
                                      strides=strides, padding="same", activation=activation)


def gray2rgb(img):
    rgb = []
    for row in img:
        n_row = []
        for pix in row:
            if type(pix) is int:
                n_row.append([pix, pix, pix])
            else:
                value = np.mean(pix)
                n_row.append([value, value, value])
        rgb.append(n_row)
    return np.array(rgb)


def save_image(filename, img, resize=None):
    img = np.clip(np.array(img), 0, 1)
    if np.shape(img)[2] == 1:
        img = gray2rgb(img)
    if resize:
        img = sk.transform.resize(img, resize)
    sk.io.imsave(filename, img)
