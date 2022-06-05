import glob
import os
import sys

import cv2
import PIL.Image
import numpy as np


GREEN = '\033[92m'
BLUE = '\033[94m'
ENDC = '\033[0m'
WARNING = '\033[93m'


def print_blue(msg):
    print(f"{BLUE}{msg}{ENDC}")


def print_green(msg):
    print(f"{GREEN}{msg}{ENDC}")


def print_warn(msg):
    print(f"{WARNING}{msg}{ENDC}")


# where the input images live
SOURCE_PATH = 'Raw'
# actually write the image to here
DESTINATION_PATH = 'Ready'
# filename prefix
PREFIX = 'img'  # default: 'img'
# if adding more images to an existing set, use this to begin numbering where you left off
START_FROM = 0
# power of two pls
SIZE = 256
# replace this with the absolute path if you're not tossing all this in the same directory
ABSOLUTE_PATH = os.getcwd()

print_blue('starting')

INPUT_DIRECTORY = f"{ABSOLUTE_PATH}\\{SOURCE_PATH}"

if not os.path.exists(INPUT_DIRECTORY):
    print_warn(f"Can't find input image directory {INPUT_DIRECTORY}!")
    print_warn("Fix that and give it another try.")
    sys.exit()

if not os.path.exists(DESTINATION_PATH):
    os.mkdir(DESTINATION_PATH)

paths = []
types = ('*.png', '*.jpg', "*.jpeg")

for ext in types:
    paths.extend(glob.glob(os.path.join(SOURCE_PATH, ext)))

print(f"{len(paths)} images found")

for i in range(len(paths)):
    idx = i + START_FROM
    new_path = f"{PREFIX}{idx:04d}.png"
    img = cv2.imread(paths[i], cv2.IMREAD_COLOR)
    try:
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
              (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
        img = np.array(PIL.Image.fromarray(img).resize((SIZE, SIZE)))

        cv2.imwrite(f"{DESTINATION_PATH}/{new_path}", img, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 100])
    except Exception as e:
        print("\n")
        print_warn(e)
        print_warn(f"error reading {paths[i]}, which was supposed to go to {idx}")
        print("\n")

print_green('done')
