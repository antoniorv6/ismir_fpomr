from tqdm import tqdm
import os
import cv2
import json
import sys
import os

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


def load_data_jsonMuret(PATH, encoding):
    X = []
    Y = []
    for file in tqdm(os.listdir(f"{PATH}")):
        markedForDelete = False
        with open(f"{PATH}/{file}") as jsonfile:
            if file.split(".")[-1] == "json":
                data = json.load(jsonfile)
                image = cv2.imread(f"{PATH}/{data['filename']}", 0)
                bbox = data["pages"][0]['bounding_box']
                image = image[bbox["fromY"]:bbox["toY"], bbox["fromX"]:bbox["toX"]]
                sequence = []
                for region in data["pages"][0]["regions"]:
                    if region["type"] == "staff":
                        if "symbols" in region: # Avoid empty staves
                            for symbol in region["symbols"]:
                                if encoding == "sseq":
                                    sequence.append(f"{symbol['agnostic_symbol_type']}")
                                    sequence.append(f"{symbol['position_in_staff']}")
                                else:
                                    sequence.append(f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}")
                if sequence:
                    X.append(image)
                    Y.append(sequence)
                else:
                    markedForDelete = True
                
        if markedForDelete:
            os.remove(f"{PATH}/{file}")
            os.remove(f"{PATH}/{data['filename']}")
                
    return X, Y


def load_data_testcase_im2s(PATH):
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
                
                Y.append(['<sos>'] + [token for token in string_array] + ['<eos>'])
    
    return X, Y


def load_data_jsonMuret_im2s(PATH):
    X = []
    Y = []
    for file in tqdm(os.listdir(f"{PATH}")):
        markedForDelete = False
        with open(f"{PATH}/{file}") as jsonfile:
            if file.split(".")[-1] == "json":
                data = json.load(jsonfile)
                image = cv2.imread(f"{PATH}/{data['filename']}", 0)
                bbox = data["pages"][0]['bounding_box']
                image = image[bbox["fromY"]:bbox["toY"], bbox["fromX"]:bbox["toX"]]
                sequence = []
                for region in data["pages"][0]["regions"]:
                    if region["type"] == "staff":
                        if "symbols" in region: # Avoid empty staves
                            for symbol in region["symbols"]:
                                sequence.append(f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}")
                if sequence:
                    X.append(image)
                    Y.append(['<sos>'] + [token for token in sequence] + ['<eos>'])
                else:
                    markedForDelete = True
                
        if markedForDelete:
            os.remove(f"{PATH}/{file}")
            os.remove(f"{PATH}/{data['filename']}")
                
    return X, Y


def load_data_muret(IMG_PATH, AGNOSTIC_PATH):
    X = []
    Y = []
    for folder in tqdm.tqdm(os.listdir(AGNOSTIC_PATH)):
        for file in os.listdir(f"{AGNOSTIC_PATH}/{folder}"):
            with open(f"{AGNOSTIC_PATH}/{folder}/{file}") as jsonfile:
                data = json.load(jsonfile)
                image = cv2.imread(f"{IMG_PATH}/{folder}/masters/{data['filename']}", 0)
                bbox = data["pages"][0]['bounding_box']
                image = image[bbox["fromY"]:bbox["toY"], bbox["fromX"]:bbox["toX"]]
                X.append(image)
                Y.append(data)
    return X, Y
