from __future__ import print_function
import cv2 as cv
import os
import glob
import sys


def load_images(folder_path):
    os.chdir(folder_path)
    image_files = glob.glob('*.jpg')
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
