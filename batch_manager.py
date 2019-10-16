"""
Author: Sigve Rokenes
Date: February, 2019

Batch manager

"""

import os
import random
import skimage as sk
from skimage import io
from skimage import transform
from util import gray2rgb
import numpy as np


class BatchManager:

    def __init__(self, path, gray=False, resize=None, limit=None, skip=0):

        self.path = path
        self.resize = resize
        self.gray = gray
        self.image_paths = os.listdir(path)
        self.images = []

        for p in self.image_paths:
            if skip > 0:
                skip -= 1
                continue
            self.images.append(self.by_name(p))
            if limit and len(self.images) == limit:
                break

        random.shuffle(self.images)
        self.batch_index = 0

    def by_name(self, name):
        img = sk.io.imread(os.path.join(self.path, name), as_gray=self.gray)
        img = sk.img_as_float(img)
        if self.resize:
            img = sk.transform.resize(img, self.resize)
        if self.gray:
            w, h = img.shape[0], img.shape[1]
            img = np.reshape(img, [w, h, 1])
        elif len(img.shape) == 2:
            img = gray2rgb(img)
        return img

    def next(self):
        return self.next_batch(1)

    def next_batch(self, batch_size=32):
        assert len(self.images) > 0, "Batch manager contains no images."
        assert batch_size <= len(self.images), "Batch size larger than data set."
        batch = []
        while len(batch) < batch_size:
            end = min(self.batch_index + batch_size, len(self.images))
            sub = self.images[self.batch_index:end-len(batch)]
            batch.extend(sub)
            self.batch_index += len(sub)

            if self.batch_index >= len(self.images):
                self.batch_index = 0
                random.shuffle(self.images)
        return batch

    def num_examples(self):
        return len(self.images)

    def sample(self, size=1):
        return random.sample(self.images, size)

