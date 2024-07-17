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
from analize_new import analize


lr = [0.0001,0.0005,0.0008,0.001,0.003]

epoch = 15

data = torch.load('data\\data12.pt')
test1 = torch.load('colored_mnist\\test1.pt')
test2 = torch.load('colored_mnist\\test2.pt')
test_data = test1+test2

acc_train = []
loss_train = []
acc_val = []
loss_val = []
acc_test = []
loss_test = []


for _ in lr :
    lr_10000 = int(_*10000)
    a,b,c,d,e,f = analize(data,test_data,epoch,_,0.75,based_model_path=None,save_path='save_model/best_model_on_new_{}.pth'.format(lr_10000),r_seed=516)  
    acc_train.append(a)
    loss_train.append(b)
    acc_val.append(c)
    loss_val.append(d)
    acc_test.append(e)
    loss_test.append(f)
   



x = np.linspace(1,epoch,epoch)

lg = []

fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,acc_train[i])
    lg.append('learning rate: '+str(lr[i]))
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Train acc')
plt.title('Acc on train')
fig1.savefig('plots/acc_ana_train.png')
fig1.clf()

fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,loss_train[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Loss on train')
fig1.savefig('plots/loss_ana_train.png')
fig1.clf()


fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,acc_val[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Validation acc')
plt.title('Acc on validation')
fig1.savefig('plots/acc_ana_val.png')
fig1.clf()

fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,loss_val[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Validation loss')
plt.title('Loss on validation')
fig1.savefig('plots/loss_ana_val.png')
fig1.clf()

fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,acc_test[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Test acc')
plt.title('Acc on test')
fig1.savefig('plots/acc_ana_test.png')
fig1.clf()

fig1 = plt.figure()
for i in range(len(lr)):
    plt.plot(x,loss_test[i])
plt.legend(lg)
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Loss on test')
fig1.savefig('plots/loss_ana_test.png')
fig1.clf()






