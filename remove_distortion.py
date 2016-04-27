from utils import load_images, display_images
from calibrate import calibrate
import cv2 as cv
import argparse


DISPLAY_SCALE = 0.45


def remove_distortion(images):
    out = calibrate(images)
    matrix = out['camera_matrix']
    dist = out['distortion_coefficient']

    undistorted_images = []
    for (image, color_image) in images:
        size = image.shape[::-1]
        new_matrix, roi = cv.getOptimalNewCameraMatrix(matrix, dist, size,
                                                       1, size)

        img = cv.undistort(color_image, matrix, dist, None, new_matrix)
        undistorted_images.append(img)

    return undistorted_images


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('folder', help='a folder wit himages\
                                             to be processed')
    args_parser.add_argument('-s', '--scale', type=float, default=0.45,
                             help='display scale to control window size')
    args = args_parser.parse_args()

    DISPLAY_SCALE = args.scale

    images = load_images(args.folder)
    undistorted_images = remove_distortion(images)
    display_images(undistorted_images, DISPLAY_SCALE)
