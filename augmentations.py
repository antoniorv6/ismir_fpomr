import cv2
import numpy as np
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective
from PIL import Image

def DPIAdjusting(sample, factor):
    width = int(np.ceil(sample.shape[1] * factor))
    height = int(np.ceil(sample.shape[0] * factor))
    return cv2.resize(sample, (width, height))

def Dilation(sample, kernel_size, iterations):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.dilate(sample, kernel=kernel, iterations=iterations)

def Erosion(sample, kernel_size, iterations):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    return cv2.erode(sample, kernel=kernel, iterations=iterations)

def Brightness(X, factor):
    X = Image.fromarray(X)
    X  = adjust_brightness(X, factor)
    return np.array(X)

def Contrast(X, factor):
    X = Image.fromarray(X)
    X = adjust_contrast(X, factor)
    return np.array(X)

def Perspective(X, scale):
    X = Image.fromarray(X)
    X = RandomPerspective(distortion_scale=scale, p=1, interpolation=Image.BILINEAR, fill=255)(X)
    return np.array(X)   


