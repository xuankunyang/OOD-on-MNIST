import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Generate the ColoredMNIST dataset using TensorFlow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the color_grayscale_arr function
def color_grayscale_arr(arr, forground_color, background_color):
    """Converts grayscale image"""
    assert arr.ndim == 2
    dtype = np.float32
    h, w = arr.shape
    arr = arr.astype(np.float32)
    arr = np.reshape(arr, [h, w, 1])
    if background_color == 'black':
        if forground_color == 'red':
            arr = arr / 255
            arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        elif forground_color == 'green':
            arr = arr / 255
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif forground_color == 'white':
            arr = arr / 255
            arr = np.concatenate([arr, arr, arr], axis=2)
    else:
        if forground_color == 'yellow':
            arr = np.concatenate([arr, arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype), arr], axis=2)

        c = [255, 255, 255]
        arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
        arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
        arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]
        arr = arr / 255
    return arr

# Create the ColoredMNIST dataset
train1_set = []
train2_set = []
train3_set = []
test1_set = []
test2_set = []
for idx, (im, label) in enumerate(zip(x_train, y_train)):
    if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(x_train)}')
    im_array = np.array(im)

    # Assign a binary label y to the image based on the digit
    binary_label = np.zeros(10)
    binary_label[label] = 1


    # Color the image according to its environment label
    if idx < 10000:
        colored_arr = color_grayscale_arr(im_array, forground_color = 'red', background_color = 'black')
        colored_arr = colored_arr * 255
        train1_set.append((colored_arr, binary_label))
    elif idx < 20000:
        colored_arr = color_grayscale_arr(im_array, forground_color = 'green', background_color = 'black')
        colored_arr = colored_arr * 255
        train2_set.append((colored_arr, binary_label))
    elif idx < 30000:
        colored_arr = color_grayscale_arr(im_array, forground_color = 'white', background_color = 'black')
        colored_arr = colored_arr * 255
        train3_set.append((colored_arr, binary_label))
    elif idx < 45000:
        colored_arr = color_grayscale_arr(im_array, forground_color = 'yellow', background_color = 'white')
        colored_arr = colored_arr * 255
        test1_set.append((colored_arr, binary_label))
    else:
        colored_arr = color_grayscale_arr(im_array, forground_color = 'blue', background_color = 'white')
        colored_arr = colored_arr * 255
        test2_set.append((colored_arr, binary_label))

# Define the directory where the PyTorch files will be saved
save_dir = 'colored_mnist'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the ColoredMNIST dataset as PyTorch files in the specified directory
torch.save(train1_set, os.path.join(save_dir, 'train1.pt'))
torch.save(train2_set, os.path.join(save_dir, 'train2.pt'))
torch.save(train3_set, os.path.join(save_dir, 'train3.pt'))
torch.save(test1_set, os.path.join(save_dir, 'test1.pt'))
torch.save(test2_set, os.path.join(save_dir, 'test2.pt'))

# Load the ColoredMNIST dataset from the PyTorch files in the specified directory
train1_set = torch.load(os.path.join(save_dir, 'train1.pt'))
train2_set = torch.load(os.path.join(save_dir, 'train2.pt'))
train3_set = torch.load(os.path.join(save_dir, 'train3.pt'))
test1_set = torch.load(os.path.join(save_dir, 'test1.pt'))
test2_set = torch.load(os.path.join(save_dir, 'test2.pt'))

# Define a function to display images from a dataset
def display_images(dataset, title):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(dataset[i][0])
        axs[i].set_title(f'{title}, label={dataset[i][1]}')
    plt.show()

# Display images from each of the 5 PyTorch files
"""display_images(train1_set, 'Train1')
display_images(train2_set, 'Train2')
display_images(train3_set, 'Train3')
display_images(test1_set, 'Test1')
display_images(test2_set, 'Test2')"""


save_dir = 'data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Combine the 8 training sets into a single training set
train_set = train1_set + train2_set + train3_set 
random.shuffle(train_set)
torch.save(train_set, os.path.join(save_dir, 'data1.pt'))
"""display_images(train_set, 'Shuffled Training Set')"""


#此部分对颜色抖动进行了更改，即是说：颜色的色调或深浅有所改变
"""
import torchvision.transforms as transforms
from PIL import Image

# Define the data augmentation transformations
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# Apply data augmentation to the training set
augmented_train_set = []
for img, label in train1_set + train2_set + train3_set:
    img = Image.fromarray(img)
    img = transform(img)
    augmented_train_set.append((np.array(img), label))

# Save the augmented training set as a PyTorch file
torch.save(augmented_train_set, os.path.join(save_dir, 'augmented_train.pt'))

# Load the augmented training set from the PyTorch file
augmented_train_set = torch.load(os.path.join(save_dir, 'augmented_train.pt'))

# Display 5 images from the augmented training set
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axs[i].imshow(augmented_train_set[i][0])
    axs[i].set_title(f'A.Train, label={augmented_train_set[i][1]}')
plt.show()
"""
