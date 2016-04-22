from __future__ import print_function
from utils import load_images
import cv2 as cv
import numpy as np
import argparse
import sys


DEBUG = False


def find_points(images):
    pattern_size = (7, 6)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((6 * 7, 3), np.float32)
    a_object_point[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Termination criteria for sub pixel corners refinement
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    print('Finding points ', end='')
    for (image, color_image) in images:
        found, corners = cv.findChessboardCorners(image, pattern_size, None)
        if found:
            obj_points.append(a_object_point)
            cv.cornerSubPix(image, corners, (11, 11), (-1, -1), stop_criteria)
            img_points.append(corners)

            print('.', end='')
        else:
            print('-', end='')

        if DEBUG:
            cv.drawChessboardCorners(color_image, pattern_size, corners, found)
            cv.imshow('corners', color_image)
            cv.waitKey(0)

        sys.stdout.flush()

    print('\nWas able to find points in %s images' % len(img_points))
    return obj_points, img_points


# images is a lis of tuples: (gray_image, color_image)
def calibrate(images):
    obj_points, img_points = find_points(images)

    if len(img_points) == 0:
        print('Impossible to calibrate: could not find any image points')
        raise

    print('Calibrating using %s images...' % len(img_points))
    image_size = images[0][0].shape[::-1]

    reprojection_error, camera_matrix, distortion_coefficient, rotation_v,\
        translation_v = cv.calibrateCamera(obj_points, img_points, image_size)

    out = {}
    out['reprojection_error'] = reprojection_error
    out['camera_matrix'] = camera_matrix
    out['distortion_coefficient'] = distortion_coefficient
    out['rotation_v'] = rotation_v
    out['translation_v'] = translation_v
    return out


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('folder', help='a folder path containing the\
                                             input JPEG images')
    args_parser.add_argument('-d', '--display-corners', action='store_true',
                             help='display resulting corners')
    args = args_parser.parse_args()

    DEBUG = args.display_corners
    images = load_images(args.folder)
    out = calibrate(images)
    print(out['camera_matrix'])
