import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Generate the ColoredMNIST dataset using TensorFlow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#调色：test  组: 1、白色背景，杂色数字(10000)  2、白色背景、彩色数字(5000)    3、黑色背景，杂色数字(10000)      4、黑色背景、彩色数字(5000)
#               5、彩色背景，杂色数字(10000)  6、彩色背景，彩色数字(10000)    7、彩色背景，黑色数字(5000)     8、彩色背景，白色数字(5000)

random_values_1 = np.random.choice([np.random.uniform(low=0, high=0.3), np.random.uniform(low=0.7, high=1)], size=(28, 28, 1))
random_values_1 = random_values_1.astype(np.float32)

def color_grayscale_arr(arr, forground_color, background_color):
    """Converts grayscale image"""
    assert arr.ndim == 2
    arr = arr.astype(np.float32)
    dtype = np.float32
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])

#白色背景
    if background_color == "white":             
        if forground_color == "various_1":      #白色背景，杂色数字(10000)
            arr = np.concatenate([ 0.3*arr*random_values_1, 0.3*arr*random_values_1, 0.4*arr], axis=2)
            c = [255, 255, 255]
            arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
            arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
            arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]
            arr = np.clip(arr,0,255)
            arr = arr / 255.0
        elif forground_color == "colored_1":    #白色背景、彩色数字(5000)
            arr = np.concatenate([ 0.4*arr, 0.6*arr, 0.8*arr], axis=2)
            c = [255, 255, 255]
            arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
            arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
            arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]
            arr = np.clip(arr,0,255)
            arr = arr / 255.0

#黑色背景
    elif background_color == "black":
        if forground_color == "various_2":      #黑色背景，杂色数字(10000)
            arr = arr / 255
            arr = np.concatenate([ random_values_1*0.8*arr, 0.6*arr, random_values_1*0.6*arr], axis=2)
        elif forground_color == "colored_2":    #黑色背景、彩色数字(5000)
            arr = arr / 255
            arr = np.concatenate([0.2*arr, 0.4*arr,0.6*arr], axis=2)

#彩色背景
    else:
        if forground_color == "various_3":      #彩色背景，杂色数字(10000)
            arr = arr / 255
            values = np.full(shape=(28, 28, 1), fill_value=0.5555, dtype=np.float32)
            arr = np.concatenate([values,random_values_1*arr,random_values_1*arr],axis = 2)
        elif forground_color == "colored_3":    #彩色背景，彩色数字(10000)
            arr = arr / 255
            arr = np.concatenate([np.ones((h,w,1),dtype = dtype), 0.4*arr,0.8*arr], axis=2)
        elif forground_color == "black":        #彩色背景，黑色数字(5000)
            arr = arr / 255
            arr_1 = np.where(arr == 0,0.6,arr)
            arr = arr_1 - arr
            arr = np.concatenate([0.2*arr,0.8*arr,np.zeros((h,w,1),dtype = dtype)], axis=2)
        elif forground_color == "white":        #彩色背景，白色数字(5000)
            arr = arr / 255
            arr_1 = np.where(arr == 0,0.6,arr)
            arr = np.concatenate([np.ones((h,w,1),dtype = dtype),arr_1,arr], axis=2)
        
    return arr



"""
    elif background_color == "blue":
        if forground_color == "green":
            arr = np.concatenate([arr,
                                  arr,
                                  arr], axis=2)
    elif background_color == "white":
        if forground_color == "black":
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                  np.zeros((h, w, 1), dtype=dtype),
                                  np.zeros((h, w, 1), dtype=dtype)], axis=2)
            
    elif background_color == "white":
        if forground_color == "various_1":
            arr = np.concatenate([2*arr , 6*arr, arr], axis=2)
        c = [255, 255, 255]
        arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
        arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
        arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]        #已完成
    
    elif background_color == "orange":
        if forground_color == "various_2":
            arr = np.concatenate([np.full((h, w, 1),285),4*arr,6*arr], axis=2)      #已完成

    return arr
"""



# Expand the ColoredMNIST dataset
train1_set_argmented = []
train2_set_argmented = []
train3_set_argmented = []
train4_set_argmented = []
train5_set_argmented = []
train6_set_argmented = []
train7_set_argmented = []
train8_set_argmented = []

for idx, (im, label) in enumerate(zip(x_train, y_train)):
    if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(x_train)}')
    im_array = np.array(im)
    im_array = im_array.astype(np.float32)

    # Assign a binary label y to the image based on the digit
    binary_label = np.zeros(10)
    binary_label[label] = 1

    # Color the image according to its environment label
    if idx < 10000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "various_1", background_color = "white")
        train1_set_argmented.append((colored_arr, binary_label))
    elif idx < 15000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "colored_1", background_color = "white")
        train2_set_argmented.append((colored_arr, binary_label))
    elif idx < 25000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "various_2", background_color = "black")
        train3_set_argmented.append((colored_arr, binary_label))
    elif idx < 30000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "colored_2", background_color = "black")
        train4_set_argmented.append((colored_arr, binary_label))
    elif idx < 40000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "various_3", background_color = "color")
        train5_set_argmented.append((colored_arr, binary_label))
    elif idx < 50000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "colored_3", background_color = "color")
        train6_set_argmented.append((colored_arr, binary_label))
    elif idx < 55000:
        colored_arr = color_grayscale_arr(im_array, forground_color = "black", background_color = "color")
        train7_set_argmented.append((colored_arr, binary_label))
    else:
        colored_arr = color_grayscale_arr(im_array, forground_color = "white", background_color = "color")
        train8_set_argmented.append((colored_arr, binary_label))

# Define the directory where the PyTorch files will be saved
save_dir = 'colored_mnist_argmented_01'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the expanded ColoredMNIST dataset as PyTorch files in the specified directory
torch.save(train1_set_argmented, os.path.join(save_dir, 'train1_argmented.pt'))
torch.save(train2_set_argmented, os.path.join(save_dir, 'train2_argmented.pt'))
torch.save(train3_set_argmented, os.path.join(save_dir, 'train3_argmented.pt'))
torch.save(train4_set_argmented, os.path.join(save_dir, 'train4_argmented.pt'))
torch.save(train5_set_argmented, os.path.join(save_dir, 'train5_argmented.pt'))
torch.save(train6_set_argmented, os.path.join(save_dir, 'train6_argmented.pt'))
torch.save(train7_set_argmented, os.path.join(save_dir, 'train7_argmented.pt'))
torch.save(train8_set_argmented, os.path.join(save_dir, 'train8_argmented.pt'))

# Load the expanded ColoredMNIST dataset from the PyTorch files in the specified directory
train1_set_argmented = torch.load(os.path.join(save_dir, 'train1_argmented.pt'))
train2_set_argmented = torch.load(os.path.join(save_dir, 'train2_argmented.pt'))
train3_set_argmented = torch.load(os.path.join(save_dir, 'train3_argmented.pt'))
train4_set_argmented = torch.load(os.path.join(save_dir, 'train4_argmented.pt'))
train5_set_argmented = torch.load(os.path.join(save_dir, 'train5_argmented.pt'))
train6_set_argmented = torch.load(os.path.join(save_dir, 'train6_argmented.pt'))
train7_set_argmented = torch.load(os.path.join(save_dir, 'train7_argmented.pt'))
train8_set_argmented = torch.load(os.path.join(save_dir, 'train8_argmented.pt'))

# Define a function to display images from a dataset
def display_images(dataset, title):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(dataset[i][0])
        axs[i].set_title(f'{title}, label={dataset[i][1]}')
    plt.show()

# Display images from each of the expanded PyTorch files
"""display_images(train1_set_argmented, 'Train1.a')
display_images(train2_set_argmented, 'Train2.a')
display_images(train3_set_argmented, 'Train3.a')
display_images(train4_set_argmented, 'Train4.a')
display_images(train5_set_argmented, 'Train5.a')
display_images(train6_set_argmented, 'Train6.a')
display_images(train7_set_argmented, 'Train7.a')
display_images(train8_set_argmented, 'Train8.a')"""



save_dir = 'data_01'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_set_argmented = train1_set_argmented + train2_set_argmented + train3_set_argmented + train4_set_argmented + train5_set_argmented + train6_set_argmented + train7_set_argmented + train8_set_argmented
random.shuffle(train_set_argmented)
torch.save(train_set_argmented, os.path.join(save_dir, 'data2.pt'))
"""display_images(train_set_argmented, 'Shuffled Training Set')"""








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