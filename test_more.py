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
from analize import analize
import matplotlib.pyplot as plt


dataset = torch.load('data\\data3.pt')
test1 = torch.load('colored_mnist\\test1.pt')
test2 = torch.load('colored_mnist\\test2.pt')

epoch = 8
lr = 0.0001
a,b,c,d = analize(dataset,epoch=epoch,lr=lr,train_ratio=0.75,based_model_path='save_model\\best_model_on_12_1.pth',save_path='save_model/best_model_on_3.pth',r_seed=77)

x = np.linspace(1,epoch,epoch)
print(a)
fig1 = plt.figure()
plt.plot(x,a)
plt.legend(['learning rate: '+str(lr)])
plt.xlabel('Epoch')
plt.ylabel('Acc')
fig1.savefig('plots/acc_on_3.png')
fig1.clf()
fig2 = plt.figure()
plt.plot(x,b)
plt.legend(['learning rate: '+str(lr)])
plt.xlabel('Epoch')
plt.ylabel('Loss')
fig2.savefig('plots/loss_on_3.png')
fig2.clf()
plot_matrix(d,c,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=lr,save=True,savepath='plots\\matrix_on_3')

device = 'cuda' if cuda.is_available() else 'cpu'
loaded_params = torch.load('save_model/best_model_on_3.pth')
net = LeNet().to(device)
net.load_state_dict(loaded_params)
loss_fun = nn.CrossEntropyLoss()

a,b,c,d = test(test1,net,loss_fun)
plot_matrix(d,c,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=lr,save=True,savepath='plots\\test1_on_3')
a,b,c,d = test(test2,net,loss_fun)
plot_matrix(d,c,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=lr,save=True,savepath='plots\\test2_on_3')