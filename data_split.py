from ast import parse
from sklearn.model_selection import train_test_split
import cv2
from utils import check_and_retrieveVocabulary, parse_arguments
import os
import tqdm
import numpy as np


def load_data(IMG_PATH, AGNOSTIC_PATH):
    X= []
    Y = []
    for file in tqdm.tqdm(os.listdir(IMG_PATH)):
        sample = file.split(".")[0]
        X.append(cv2.imread(f"{IMG_PATH}{sample}.png", 0))
        with open(f"{AGNOSTIC_PATH}{sample}.txt") as f:
            string_array = f.readline().split("+")
            for idx, token in enumerate(string_array):
                string_array[idx] = token.strip()
            
            Y.append(string_array)
    
    return X, Y

def main():
    args = parse_arguments()
    ratio = 0.5
    X, Y = load_data(IMG_PATH=args.image_folder, AGNOSTIC_PATH=args.agnostic_folder)

    for i in range(len(X)):
        #img = (255. - X[i]) / 255.
        img = X[i]
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        X[i] = cv2.resize(img, (width, height))

    


if __name__ == "__main__":
    main()
