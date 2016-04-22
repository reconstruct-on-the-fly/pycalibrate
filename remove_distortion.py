from utils import load_images
from calibrate import calibrate
import cv2 as cv
import argparse


def remove_distortion(images):
    out = calibrate(images)
    matrix = out['camera_matrix']
    dist = out['distortion_coefficient']

    for (image, color_image) in images:
        size = image.shape[::-1]
        new_matrix, roi = cv.getOptimalNewCameraMatrix(matrix, dist, size, 1,
                                                       size)

        out_image = cv.undistort(image, matrix, dist, None, new_matrix)
        cv.imshow('Undistorted images', out_image)
        cv.waitKey(0)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('folder', help='a folder wit himages\
                                             to be processed')
    args = args_parser.parse_args()

    images = load_images(args.folder)
    remove_distortion(images)
