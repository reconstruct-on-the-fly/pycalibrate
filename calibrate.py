from __future__ import print_function
import cv2 as cv
import argparse, sys, os, glob

def load_images(folder_path):
    os.chdir(folder_path)
    image_files = glob.glob('*.jpg');
    print('Found %s images' % len(image_files))
    if len(image_files) == 0: return

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



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("folder", help="a folder path containing the\
                                             input JPEG images")
    args = args_parser.parse_args()

    load_images(args.folder)
