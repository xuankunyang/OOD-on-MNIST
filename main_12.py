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

dataset = torch.load('data\\data12.pt')

np.random.seed = 516
num = len(dataset)
np.random.shuffle(dataset)
train_num =int(num*0.75)
X_train = dataset[:train_num]
X_val = dataset[train_num:]
x_test1 = torch.load('colored_mnist\\test1.pt')
x_test2 = torch.load('colored_mnist\\test2.pt')

device = 'cuda' if cuda.is_available() else 'cpu'
learning_rate = 0.001
net = LeNet().to(device)
#损失函数：
loss_fun = nn.CrossEntropyLoss()
#梯度下降优化器
optimizer = SGD(net.parameters(),lr=learning_rate,momentum=0.9)
#梯度优化
lr_updater = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


epoch = 15
acc = []
loss = []
for t in track(range(epoch),description="进度..."):
    print("epoch {}:".format(t+1))
    loss_,acc_ = train(net,loss_fun,X_train,optimizer)
    loss.append(loss_)
    acc.append(acc_)
    
folder = 'save_model'
if not os.path.exists(folder):
    os.mkdir('save_model')
print('save best model')
torch.save(net.state_dict(), 'save_model/best_model_on_12.pth')
acc1,loss1,y_pred1,y_true1 = test(X_val,net,loss_fun)
acc2,loss2,y_pred2,y_true2 = test(x_test1,net,loss_fun)
acc3,loss3,y_pred3,y_true3 = test(x_test2,net,loss_fun)

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
plt.savefig('plots\\train_on_12')
fig.clf()

plot_matrix(y_true1,y_pred1,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=learning_rate,savepath='plots\\val_on_12',save=True)
plot_matrix(y_true2,y_pred2,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=learning_rate,savepath='plots\\test1_on_12',save=True)
plot_matrix(y_true3,y_pred3,[0,1,2,3,4,5,6,7,8,9], title='confusion_matrix',axis_labels=[0,1,2,3,4,5,6,7,8,9],lr=learning_rate,savepath='plots\\test2_on_12',save=True)