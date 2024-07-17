import os
import torch
import numpy as np
from torch import nn,cuda
from NN_class import LeNet
from torch.optim import SGD,lr_scheduler,Adam
from train import train
from test import test
from rich.progress import track
from plot_metrics import plot_matrix
from torchvision import transforms
from rich.progress import track
from sklearn.metrics import accuracy_score

data = torch.load('data\\data12.pt')

print(np.max(data[8][0]))
print(np.min(data[8][0]))