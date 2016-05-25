from __future__ import print_function
from utils import load_images, display_images, ndarray_to_str
import cv2 as cv
import numpy as np
import argparse
import sys
import os


DEBUG = False
DISPLAY_SCALE = 0.45
OUTPUT_FILE = os.getcwd() + '/output.txt'
PATTERN_SIZE = (9, 6)


def find_points(images):
    pattern_size = (9, 6)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3),
                              np.float32)
    a_object_point[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                                     0:PATTERN_SIZE[1]].T.reshape(-1, 2)

    # Termination criteria for sub pixel corners refinement
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    print('Finding points ', end='')
    debug_images = []
    for (image, color_image) in images:
        found, corners = cv.findChessboardCorners(image, PATTERN_SIZE, None)
        if found:
            obj_points.append(a_object_point)
            cv.cornerSubPix(image, corners, (11, 11), (-1, -1), stop_criteria)
            img_points.append(corners)

            print('.', end='')
        else:
            print('-', end='')

        if DEBUG:
            cv.drawChessboardCorners(color_image, PATTERN_SIZE, corners, found)
            debug_images.append(color_image)

        sys.stdout.flush()

    if DEBUG:
        display_images(debug_images, DISPLAY_SCALE)

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
    args_parser.add_argument('-s', '--scale', type=float, default=0.45,
                             help='display scale to control window size')
    args_parser.add_argument('-o', '--out', default=OUTPUT_FILE,
                             help='absolute path for output file')
    args_parser.add_argument('-p', '--pattern', default="9X6",
                             help='WXH pattern size for chessboard pictures')
    args = args_parser.parse_args()

    DEBUG = args.display_corners
    DISPLAY_SCALE = args.scale
    OUTPUT_FILE = args.out
    PATTERN_SIZE = tuple(map(int, args.pattern.split('X')))

    images = load_images(args.folder)
    out = calibrate(images)
    print('Calubration results saved in %s' % OUTPUT_FILE)

    with open(OUTPUT_FILE, 'w') as file:
        file.write(ndarray_to_str(out['camera_matrix']))
        file.write(ndarray_to_str('\n\n'))
        file.write(ndarray_to_str(out['distortion_coefficient']))
        file.write(ndarray_to_str('\n\n'))
        file.write(ndarray_to_str(out['reprojection_error']))
