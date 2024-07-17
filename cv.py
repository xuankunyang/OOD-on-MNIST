import cv2
import numpy as np
import random

def add_noise(image):
    # Choose a random noise type
    noise_type = random.choice(['gaussian', 'poisson', 'speckle', 's&p'])

    if noise_type == "gaussian":
        row,col,ch= image.shape
        mean = 0
        var = 0.001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 1)
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 1)
    elif noise_type == "speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch) * 0.05
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return np.clip(noisy, 0, 1)
    else:
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.0001
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

def augment_image(image):
    # Rotate the image
    angle = np.random.uniform(-10, 10)
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    rotated_mat = cv2.warpAffine(image, rotation_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # Add noise to the image
    noisy_image = add_noise(rotated_mat)
    noisy_image = noisy_image.astype(np.float32)
    return noisy_image