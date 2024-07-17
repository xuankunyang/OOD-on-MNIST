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

def analize(dataset,epoch,lr,train_ratio,based_model_path='save_model\\best_model.pth',save_path='save_model/best_model_test.pth',r_seed=77):
    np.random.seed = r_seed
    num = len(dataset)
    np.random.shuffle(dataset)
    train_num =int(num*train_ratio)
    X_train = dataset[:train_num]
    X_test = dataset[train_num:]
    device = 'cuda' if cuda.is_available() else 'cpu'

    if based_model_path is None:
        net = LeNet().to(device)
    else:
        loaded_params = torch.load(based_model_path)
        net = LeNet().to(device)
        net.load_state_dict(loaded_params)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(),lr=lr,momentum=0.9)
    lr_updater = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    acc = []
    loss = []
    for t in track(range(epoch),description="进度..."):
        print("epoch {}:".format(t+1))
        loss_1,acc_1 = train(net,loss_fun,X_train,optimizer)
        loss.append(loss_1)
        acc.append(acc_1)
    print('save best model')
    
    torch.save(net.state_dict(), save_path)
    loss_2,acc_2,y_pred,y_true = test(X_test,net,loss_fun)
    
    return acc,loss,y_pred,y_true