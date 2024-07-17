import os
import torch
import numpy as np
from torch import nn,cuda
from NN_class import LeNet
from torch.optim import SGD,lr_scheduler,Adam
from train import train
from test import test
from rich.progress import track
import matplotlib as mlt 
import matplotlib.pyplot as plt
from plot_metrics import plot_matrix
from NN_class import LeNet
from sklearn.model_selection import train_test_split
from analize import analize


lr = [0.0001,0.0005,0.0008,0.001]

epoch = 8

data = torch.load('data\\data12.pt')
test1 = torch.load('colored_mnist\\test1.pt')
test2 = torch.load('colored_mnist\\test2.pt')
test_data = test1+test2

acc = []
loss = []
test_acc = []

for _ in lr :
    lr_10000 = int(_*10000)
    a,b,c,d = analize(data,epoch,_,0.75,based_model_path='save_model/best_model_on_12.pth',save_path='save_model/best_model_on_12_{}.pth'.format(lr_10000),r_seed=516)  
    acc.append(a)
    loss.append(b)
    plot_matrix(d,c,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=_,save=True,savepath='plots\\val_ana_on_12_{}'.format(lr_10000))
    device = 'cuda' if cuda.is_available() else 'cpu'
    loaded_params = torch.load('save_model/best_model_on_12_{}.pth'.format(lr_10000))
    net = LeNet().to(device)
    net.load_state_dict(loaded_params)
    loss_fun = nn.CrossEntropyLoss()
    a,b,c,d = test(test_data,net,loss_fun)
    test_acc.append(b)



x = np.linspace(1,epoch,epoch)
fig1 = plt.figure()
lg = []
for i in range(len(lr)):
    plt.plot(x,acc[i])
    lg.append('learning rate: '+str(lr[i]))
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Acc')
fig1.savefig('plots/acc_ana_on_12.png')
fig1.clf()
fig2 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,loss[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Loss')
fig2.savefig('plots/loss_ana_on_12.png')
fig2.clf()
fig3 = plt.figure()
plt.plot(lr,test_acc)
plt.xlabel('Learning rate')
plt.ylabel('Test acc')
fig3.savefig('plots\\test_acc_ana_on_12.png')
fig3.clf()








