from __future__ import print_function
from math import sqrt, floor, ceil
import cv2 as cv
import numpy as np
import os
import glob
import sys
import re


def load_images(folder_path):
    os.chdir(folder_path)
    image_files = glob.glob('*.JPG')
    print('Found %s images' % len(image_files))
    if len(image_files) == 0:
        return

    images = []
    print('Loading images ', end='')
    for file in image_files:
        image = cv.imread(file)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images.append((gray_image, image))
        print('.', end='')
        sys.stdout.flush()

    print('')
    return images


def display_images(images, scale_factor):
    height, width = images[0].shape[:2]
    rows = int(floor(sqrt(len(images))))
    cols = int(ceil(len(images) / float(rows)))

    out_size = (rows * height, cols * width, 3)
    out_image = np.full(out_size, 255, dtype=np.uint8)

    # For every image, copy to output
    for i in range(rows):
        for j in range(cols):
            k = (i * cols) + j
            # only if we have enough images
            if k < len(images):
                new_row = i * height
                new_col = j * width
                out_image[new_row:(new_row + height),
                          new_col:(new_col + width)] = images[k]

    resized_out = cv.resize(out_image, (0, 0), fx=scale_factor,
                            fy=scale_factor)

    cv.imshow('Object points', resized_out)
    cv.waitKey(0)


def ndarray_to_str(ndarray):
    ndarray_str = str(ndarray)
    ndarray_str = ndarray_str.replace('[', '')
    ndarray_str = ndarray_str.replace(']', '')
    ndarray_str = re.sub(' +', ' ', ndarray_str)
    return ndarray_str
