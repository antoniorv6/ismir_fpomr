from sklearn.model_selection import train_test_split
import cv2
from utils import check_and_retrieveVocabulary, parse_arguments_ds
import os
import tqdm
import numpy as np
import sys
import json

def load_data_muret(IMG_PATH, AGNOSTIC_PATH):
    X = []
    Y = []
    for folder in tqdm.tqdm(os.listdir(AGNOSTIC_PATH)):
        for file in os.listdir(f"{AGNOSTIC_PATH}/{folder}"):
            with open(f"{AGNOSTIC_PATH}/{folder}/{file}") as jsonfile:
                data = json.load(jsonfile)
                image = cv2.imread(f"{IMG_PATH}/{folder}/masters/{data['filename']}")
                sequence = []
                for region in data["pages"][0]["regions"]:
                    if region["type"] == "staff":
                        if "symbols" in region: # Avoid empty staves
                            for symbol in region["symbols"]:
                                sequence.append(f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}")
                if sequence:
                    X.append(image)
                    Y.append(sequence)
    return X, Y

def load_data(IMG_PATH, AGNOSTIC_PATH):
    X= []
    Y = []
    for file in tqdm.tqdm(os.listdir(IMG_PATH)):
        sample = file.split(".")[0]
        X.append(cv2.imread(f"{IMG_PATH}{sample}.png"))
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
        cv2.imwrite(f"{path}/{idx}.png", sample)
        with open(f"{path}/{idx}.txt", "w+") as wfile:
            wfile.write(" ".join(Y[idx]))

def save_partition_json(corpus_name,folder, X, Y):
    path = f"Data/{corpus_name}/{folder}"
    if not os.path.isdir(f"Data/{corpus_name}/{folder}"):
        os.makedirs(path)

    for idx, sample in enumerate(X):
        filename = Y[idx]["filename"]
        cv2.imwrite(f"{path}/{filename}", sample)
        with open(f"{path}/{filename}.json", "w+") as wfile:
            json.dump(Y[idx], wfile)

def main():
    args = parse_arguments_ds()
    ratio = 0.5
    X, Y = load_data_muret(IMG_PATH=args.image_folder, AGNOSTIC_PATH=args.agnostic_folder)

    XTrain, XValTest, YTrain, YValTest = train_test_split(X,Y, test_size=0.3, shuffle=True)

    XVal, XTest, YVal, YTest = train_test_split(XValTest,YValTest, test_size=0.5)

    save_partition_json(args.corpus_name, "train", XTrain, YTrain)
    save_partition_json(args.corpus_name, "val", XVal, YVal)
    save_partition_json(args.corpus_name, "test", XTest, YTest)



if __name__ == "__main__":
    main()
