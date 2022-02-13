from tqdm import tqdm
import os
import cv2

def load_data_testcase(PATH):
    X= []
    Y = []
    for file in tqdm(os.listdir(PATH)):
        sample = file.split(".")[0]
        format = file.split(".")[1]
        if format == "png":
            X.append(cv2.imread(f"{PATH}{sample}.png", 0))
            with open(f"{PATH}{sample}.txt", "r") as agnosticfile:
                string_array = agnosticfile.readline().split(" ")
                for idx, token in enumerate(string_array):
                    string_array[idx] = token.strip()
                    Y.append(string_array)
    
    return X, Y