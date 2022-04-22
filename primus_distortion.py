import os
from wand.image import Image
from wand.color import Color
import random
import shutil
import sys
import tqdm

def main():
    if not os.path.isdir("Data/Camera_FP-Primus"):
        os.makedirs("Data/Camera_FP-Primus/train")
        os.makedirs("Data/Camera_FP-Primus/val")
        os.makedirs("Data/Camera_FP-Primus/test")



    for file in tqdm.tqdm(os.listdir("Data/FP-Primus/test")):
        if file.endswith(".png"):
            im = Image(filename=f"Data/FP-Primus/test/{file}")
            im.chop(int(random.randint(1, 5)), int(random.randint(1, 6)), int(random.randint(1, 300)), int(random.randint(1, 50)))
            im.swirl(round(random.uniform(-3.00, 3.00), 2))
            im.spread(0.1)
            im.shear("WHITE", round(random.uniform(-5.00, 5.00), 2), round(random.uniform(-1.50, 1.50), 2))
            im.wave(round(random.uniform(0.00, 0.50), 2), round(random.uniform(0.00, 0.40), 2))
            im.rotate(round(random.uniform(0.00, 0.30), 2))
            im.noise("gaussian", round(random.uniform(0.00, 1.25), 2))
            im.wave(round(random.uniform(0.00, 0.50), 2), round(random.uniform(0.00, 0.40), 2))
            im.motion_blur(radius=round(random.uniform(-7.00, 5.00), 2), sigma=round(random.uniform(-7.00, 7.00), 2), angle=round(random.uniform(-7.00, 6.00), 2))
            im.format = 'jpeg'
            im.save(filename=f"Data/Camera_FP-Primus/test/{file.split('.')[0]}.jpg")

        else:
            shutil.copy(f"Data/FP-Primus/test/{file}", f"Data/Camera_FP-Primus/test/{file}")



    pass

if __name__=="__main__":
    main()
