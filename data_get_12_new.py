import torch
import random
import os
import matplotlib.pyplot as plt
import numpy as np

# Define a function to display images from a dataset
def display_images(dataset, title):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(dataset[i][0])
        axs[i].set_title(f'{title}, label={dataset[i][1]}')
    plt.show()

save_dir = 'data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the other dataset
data = torch.load('data\\data1.pt')
data_argmented = torch.load('data\\data2.pt')

# Combine the datasets
combined_dataset = data + data_argmented

random.shuffle(combined_dataset)
torch.save(combined_dataset, os.path.join(save_dir, 'data12.pt'))
"""display_images(combined_dataset, 'Shuffled Training Set')"""