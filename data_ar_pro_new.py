import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from cv import add_noise,augment_image

save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_set_argmented_pro_load = torch.load('data_01\\data2.pt')
train_set_argmented_pro = []
# Augment the train_set_argmented dataset
for image, label in train_set_argmented_pro_load:
    image = np.array(image)
    augmented_image = augment_image(image)
    augmented_image = augmented_image * 255
    train_set_argmented_pro.append((augmented_image, label))

random.shuffle(train_set_argmented_pro)
# Save the augmented dataset
torch.save(train_set_argmented_pro, os.path.join(save_dir, 'data3.pt'))
# Load the augmented training set from the PyTorch file
argmented_dataset = torch.load(os.path.join(save_dir, 'data3.pt'))

# Display 5 images from the augmented training set
"""fig, axs = plt.subplots(1, 10, figsize=(10, 2))
for i in range(10):
    axs[i].imshow(argmented_dataset[i][0])
    axs[i].set_title(f'A.Train, label={argmented_dataset[i][1]}')
plt.show()"""