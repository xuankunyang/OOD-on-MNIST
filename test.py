import os
import torch
import numpy as np
from torch import nn,cuda
from NN_class import LeNet
from sklearn.metrics import accuracy_score
from tqdm import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'

def test(dataloader,model,loss_fun):
    model.eval()#防止改变权值
    with torch.no_grad():
        total_loss = 0
        num = 0
        y_pred = []
        y_true = []
        for n,(x,Y) in enumerate(tqdm(dataloader,desc="进度...")):
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
            y_pred.append(y1.cpu().numpy())
            y_true.append(y2.cpu().numpy())

        test_loss = total_loss/num
        test_loss = test_loss.cpu().detach().numpy()
        test_acc = accuracy_score(y_true,y_pred)
        print("test loss:"+str(test_loss))
        print("test acc:"+str(test_acc))

    return test_loss,test_acc,y_pred,y_true

