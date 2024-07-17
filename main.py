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

train_dataloader1 = torch.load('colored_mnist\\train1.pt')
train_dataloader2 = torch.load('colored_mnist\\train2.pt')
train_dataloader3 = torch.load('colored_mnist\\train3.pt')
test_dataloader1 = torch.load('colored_mnist\\test1.pt')
test_dataloader2 = torch.load('colored_mnist\\test2.pt')

device = 'cuda' if cuda.is_available() else 'cpu'

learning_rate = 0.001
net = LeNet().to(device)
#损失函数：
loss_fun = nn.CrossEntropyLoss()
#梯度下降优化器
optimizer = SGD(net.parameters(),lr=learning_rate,momentum=0.9)
#梯度优化
lr_updater = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


epoch = 15
acc = []
loss = []
for t in track(range(epoch),description="进度..."):
    print("epoch {}:".format(t+1))
    loss1,acc1 = train(net,loss_fun,train_dataloader1,optimizer)
    loss2,acc2 = train(net,loss_fun,train_dataloader2,optimizer)
    loss3,acc3 = train(net,loss_fun,train_dataloader3,optimizer)
    loss.append((loss1+loss2+loss3)/3)
    acc.append((acc1+acc2+acc3)/3)
    
folder = 'save_model'
if not os.path.exists(folder):
    os.mkdir('save_model')
print('save best model')
torch.save(net.state_dict(), 'save_model/best_model.pth')
acc1,loss1,y_pred1,y_true1 = test(test_dataloader1,net,loss_fun)
acc2,loss2,y_pred2,y_true2 = test(test_dataloader2,net,loss_fun)

x = np.linspace(1,epoch,epoch)
fig =plt.figure()
plt.subplot(121)
plt.plot(x,acc)
plt.xlabel("Epoch")
plt.ylabel("Average Acc")
plt.text(x=1,y=0.97,s="learning rate: {}".format(learning_rate))
plt.subplot(122)
plt.plot(x,loss)
plt.xlabel("Epoch")
plt.ylabel("Average Cross Entropy Loss")
plt.text(x=1.6,y=1.2,s="learning rate: {}".format(learning_rate))
plt.savefig('plots\\basic_train')
plt.clf()
plot_matrix(y_true1,y_pred1,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=learning_rate)
plot_matrix(y_true2,y_pred2,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=learning_rate)





