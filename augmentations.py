import cv2
import numpy as np
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective, RandomCrop, ColorJitter

def DPIAdjusting(sample, factor):
    width = int(np.ceil(sample.shape[1] * factor))
    height = int(np.ceil(sample.shape[0] * factor))
    return cv2.resize(sample, (width, height))

def Dilation(sample, kernel_size, iterations):
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    return cv2.dilate(sample, kernel=kernel, iterations=iterations)

def Erosion(sample, kernel_size, iterations):
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    return cv2.erode(sample, kernel=kernel, iterations=iterations)

def Brightness(X, factor):
    return adjust_brightness(X, factor)

def Contrast(X, factor):
    return adjust_contrast(X, factor)

def SignFlipping(X):
    return SignFlipping()(X)

def RandomTransform(X, max_value):
    return RandomTransform(max_value=max_value)(X)   


