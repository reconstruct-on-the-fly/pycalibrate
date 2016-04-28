from __future__ import print_function
from utils import load_images, display_images
import cv2 as cv
import numpy as np
import yaml
import argparse
import sys, os


DEBUG = False
DISPLAY_SCALE = 0.45
OUTPUT_FILE = os.getcwd() + '/output.yml'


def find_points(images):
    pattern_size = (9, 6)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((6 * 9, 3), np.float32)
    a_object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Termination criteria for sub pixel corners refinement
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    print('Finding points ', end='')
    debug_images = []
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
    matrix_to_str(camera_matrix)
    return out

def matrix_to_str(matrix):
    matrix_str = str(matrix)
    matrix_str = matrix_str.replace('[', '')
    matrix_str = matrix_str.replace(']', '')
    print(matrix_str)
    return matrix_str
        


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('folder', help='a folder path containing the\
                                             input JPEG images')
    args_parser.add_argument('-d', '--display-corners', action='store_true',
                             help='display resulting corners')
    args_parser.add_argument('-s', '--scale', type=float, default=0.45,
                             help='display scale to control window size')
    args = args_parser.parse_args()

    DEBUG = args.display_corners
    DISPLAY_SCALE = args.scale

    images = load_images(args.folder)
    out = calibrate(images)
    print('Calubration results saved in %s' % OUTPUT_FILE)

    with open(OUTPUT_FILE, 'w') as file:
        file.write(yaml.dump(out))
