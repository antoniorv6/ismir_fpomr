from sklearn.model_selection import train_test_split
import cv2
from utils import check_and_retrieveVocabulary, parse_arguments_ds
import os
import tqdm
import numpy as np
import sys
import json

def load_data_muret(IMG_PATH, JSON_PATH):
    X = []
    Y = []
    for folder in tqdm.tqdm(os.listdir(JSON_PATH)):
        for file in os.listdir(f"{JSON_PATH}/{folder}"):
            with open(f"{JSON_PATH}/{folder}/{file}") as jsonfile:
                data = json.load(jsonfile)
                print(data)
                sys.exit(0)

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

def save_partition(corpus_name,folder, X, Y):
    path = f"Data/{corpus_name}/{folder}"
    if not os.path.isdir(f"Data/{corpus_name}/{folder}"):
        os.makedirs(path)

    for idx, sample in enumerate(X):
        cv2.imwrite(f"{path}/{idx}.jpg", sample)
        with open(f"{path}/{idx}.txt", "w+") as wfile:
            wfile.write(" ".join(Y[idx]))

def main():
    args = parse_arguments_ds()
    ratio = 0.5
    X, Y = load_data_muret(IMG_PATH=args.image_folder, JSON_PATH=args.agnostic_folder)
    _,_ = check_and_retrieveVocabulary(Y, f"./vocab", f"{args.corpus_name}")

    for i in range(len(X)):
        #img = (255. - X[i]) / 255.
        img = X[i]
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        X[i] = cv2.resize(img, (width, height))

    XTrain, XValTest, YTrain, YValTest = train_test_split(X,Y, test_size=0.3, shuffle=True)

    XVal, XTest, YVal, YTest = train_test_split(XValTest,YValTest, test_size=0.5)

    save_partition(args.corpus_name, "train", XTrain, YTrain)
    save_partition(args.corpus_name, "val", XVal, YVal)
    save_partition(args.corpus_name, "test", XTest, YTest)



if __name__ == "__main__":
    main()
