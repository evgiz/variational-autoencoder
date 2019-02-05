"""
Autor: Sigve Rokenes
Date: February, 2019

Data preparation

"""

import os
import random
import skimage as sk
from skimage import io
from skimage import transform


def augment(image):
    augments = []
    augments_per_image = 10
    for i in range(augments_per_image):
        img = sk.img_as_float(image, force_copy=True)
        if i < augments_per_image/2:
            img = img[:, ::-1]
        rnd_angle = random.uniform(-10, 10)
        img = sk.transform.rotate(img, rnd_angle, mode='edge')
        augments.append(img)
    return augments


if __name__ == "__main__":

    save_path = "data/processed/"
    root = "data/raw"

    image_index = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    image_paths = os.listdir(root)
    print("Processing {} images".format(len(image_paths)))

    for ip in image_paths:
        original = sk.io.imread(os.path.join(root, ip))
        augments = augment(original)

        for img in range(len(augments)):
            name = "{:05d}.png".format(image_index)
            sk.io.imsave(os.path.join(save_path, name), augments[img])
            image_index += 1

    print("All done")