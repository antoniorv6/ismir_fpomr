
from DataAugmentationGenerator import DataAugmentationGenerator
import cv2
import random
import argparse


def menu():
    parser = argparse.ArgumentParser(description='DA SAE')
    parser.add_argument('-n',   default=100, type=int,  help='Number of images to be generated')
    parser.add_argument('-pages',   default=0, type=int,  help='Number of real pages to be considered')
    parser.add_argument('-jsons', type=str, default=None,  help='Path to the GT dataset directory (json)')
    parser.add_argument('-txt_train', type=str,  default=None, help='Path to the txt file with the paths of the images.')
    parser.add_argument('--vrs',   action='store_true', help='Active the vertical resize of regions')
    parser.add_argument('--folds',   action='store_true', help='Generate folds')
    parser.add_argument('--uniform_rotate',   action='store_true', help='Uniform the rotation for each page')
    parser.add_argument('-seed',   default=42, type=int,  help='Seed')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


if __name__ == "__main__":
    config = menu()

    random.seed(config.seed)

    jsons = config.jsons
    fold = None
    parent_dir_str = None
    DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelectionAuto(jsons, config.n, config.uniform_rotate, config.vrs)
    