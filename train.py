import os
import torch
import numpy as np
from torch import nn,cuda
from NN_class import LeNet
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = 'cuda' if cuda.is_available() else 'cpu'

def train(model,loss_fun,dataloader,optimizer):
    total_loss = 0
    num = 0
    correct = 0
    for n,(x,Y) in enumerate(tqdm(dataloader,desc="进度...")):
        #clear
        optimizer.zero_grad()
        #forward
        x = torch.tensor(x)
        Y = torch.tensor(Y)
        x = x.permute(2,1,0)
        x,Y = x.to(device),Y.to(device)
        output = model(x)
        output = torch.squeeze(output)
        #num
        num+=1
        #loss
        loss = loss_fun(output,Y)
        total_loss+=loss
        #argmax
        y1 = torch.argmax(output)
        y2 = torch.argmax(Y)
        if(y1==y2):
            correct+=1
        #calculate gradient
        loss.backward()
        #update
        optimizer.step()

    train_loss =(total_loss/num)
    train_loss = train_loss.cpu().detach().numpy()
    train_acc = correct/num
    print("train loss:"+str(train_loss))
    print("train acc:"+str(train_acc))

    return train_loss,train_acc
    


        

    