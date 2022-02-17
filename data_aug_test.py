from augmentations import Brightness, Contrast, DPIAdjusting, Dilation, Erosion, Perspective
import numpy as np
import cv2

def augmentation_process(X):

    X = np.array(X)

    if np.random.rand() < 0.2:
        print("DPI")
        scale = np.random.uniform(0.75, 1)
        X = DPIAdjusting(X, scale)
    
    if np.random.rand() < 0.2:
        kernel_size = np.random.randint(1, 3)
        iterations = 1
        print(f"Dilation - {kernel_size}")
        X = Dilation(X, kernel_size, iterations)
    
    if np.random.rand() < 0.2:
        kernel_size = np.random.randint(1, 3)
        iterations = 1
        print(f"Erosion - {kernel_size}")
        X = Erosion(X, kernel_size, iterations)
    
    if np.random.rand() < 0.2:
        brightness_factor = np.random.uniform(0.01, 1)
        print(f"Brightness - {brightness_factor}")
        X = Brightness(X, brightness_factor)
    
    if np.random.rand() < 0.2:
        contrast_factor = np.random.uniform(0.01, 1)
        print(f"Contrast - {contrast_factor}")
        X = Contrast(X, contrast_factor)
    
    if np.random.rand() < 0.2:
        scale_factor = np.random.uniform(0, 0.3)
        print(f"Random perspective - {scale_factor}")
        X = Perspective(X, scale_factor)

    return X

def main():
    image = cv2.imread("test_source.jpg", 0)
    X = augmentation_process(image)
    X = (255. - X) / 255.
    
    cv2.imwrite("test_augmented.jpg", 255. * X)

if __name__ == "__main__":
    main()
