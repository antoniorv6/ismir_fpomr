import os
import cv2
import numpy as np

from utils import parse_arguments, check_and_retrieveVocabulary


def load_data(IMG_PATH, AGNOSTIC_PATH):
    X= []
    Y = []
    for file in os.listdir(IMG_PATH):
        sample = file.split(".")
        X.append(cv2.imread(f"{IMG_PATH}{sample}.png"))
        with open()
        X.append(cv2.imread(f"{IMG_PATH}{file}.png"))

        

def main():
    args = parse_arguments()
    load_data(IMG_PATH=args.image_folder, AGNOSTIC_PATH=args.agnostic_folder)


if __name__=="__main__":
    main()
