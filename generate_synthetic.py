from DataAugmentationGenerator import DataAugmentationGenerator
import random
import cv2
import json
import os
import tqdm

def main():
    random.seed(42)
    os.makedirs("Data/SEILS/train_daug")
    for i in tqdm.tqdm(range(2000)):
        image, json_str = DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelectionAuto("Data/SEILS/train/", 1, False, False, 0.2)
        json_str = json.loads(json_str)
        cv2.imwrite(f"Data/SEILS/train_daug/{i}.png", image)
        with open(f"Data/SEILS/train_daug/{i}.json", "w") as jsonfile:
            json.dump(json_str, jsonfile)
        
    pass

if __name__ == "__main__":
    main()
