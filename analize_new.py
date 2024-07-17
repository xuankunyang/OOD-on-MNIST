import os
import torch
import numpy as np
from torch import nn,cuda
from NN_class import LeNet
from torch.optim import SGD
from train import train
from test import test
from rich.progress import track
import matplotlib as mlt 
import matplotlib.pyplot as plt
from plot_metrics import plot_matrix
from NN_class import LeNet
from sklearn.model_selection import train_test_split

def analize(dataset,testset,epoch,lr,train_ratio,based_model_path='save_model\\best_model.pth',save_path='save_model/best_model_test.pth',r_seed=77):
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
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    acc_test = []
    loss_test = []
    for t in track(range(epoch),description="进度..."):
        print("epoch {}:".format(t+1))
        loss_1,acc_1 = train(net,loss_fun,X_train,optimizer)
        loss_train.append(loss_1)
        acc_train.append(acc_1)
        loss_2,acc_2,y_pred,y_true = test(X_test,net,loss_fun)
        loss_val.append(loss_2)
        acc_val.append(acc_2)
        loss_3,acc_3,y_pred,y_true = test(testset,net,loss_fun)
        loss_test.append(loss_3)
        acc_test.append(acc_3)
        

    print('save best model')
    torch.save(net.state_dict(), save_path)

    return acc_train,loss_train,acc_val,loss_val,acc_test,loss_test

