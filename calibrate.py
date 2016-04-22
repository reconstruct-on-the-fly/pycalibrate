from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import sys
import os
import glob


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
        images.append(gray_image)
        print('.', end='')
        sys.stdout.flush()

    print("")
    return images


def find_points(images):
    pattern_size = (6, 7)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((6 * 7, 3), np.float32)
    a_object_point[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Termination criteria for sub pixel corners refinement
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    print('Finding points ', end='')
    for image in images:
        found, corners = cv.findChessboardCorners(image, pattern_size, None)
        if found:
            obj_points.append(a_object_point)
            cv.cornerSubPix(image, corners, (11, 11), (-1, -1), stop_criteria)
            img_points.append(corners)
            print('.', end='')
        else:
            print('-', end='')
        sys.stdout.flush()

    print('\nWas able to find points in %s images' % len(img_points))
    return obj_points, img_points


def calibrate(folder_path):
    images = load_images(folder_path)
    obj_points, img_points = find_points(images)

    if len(img_points) == 0:
        print('Impossible to calibrate: could not find any image points')
        raise

    print('Calibrating using %s images...' % len(img_points)) 
    image_size = images[0].shape[::-1]

    reprojection_error, camera_matrix, distortion_coefficient, rotation_v,\
    translation_v = cv.calibrateCamera(obj_points, img_points, image_size)

    print(camera_matrix)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("folder", help="a folder path containing the\
                                             input JPEG images")
    args = args_parser.parse_args()
    calibrate(args.folder)
