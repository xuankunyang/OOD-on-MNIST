import numpy as np
import torch
from torch import cuda,nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.kernel_size = [5,5,5]
        self.pooling_size = [2,2]
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=self.kernel_size[0],padding=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=self.kernel_size[1])
        self.active_fun = nn.Sigmoid()
        self.pooling1 = nn.AvgPool2d(kernel_size=self.pooling_size[0],stride=self.pooling_size[0])
        self.pooling2 = nn.AvgPool2d(kernel_size=self.pooling_size[1],stride=self.pooling_size[1])
        self.fully_connect1 = nn.Linear(400,120)
        self.fully_connect2 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)
    
    def forward(self,x):
        x = self.active_fun(self.conv1(x))
        x = self.pooling1(x)
        x = self.active_fun(self.conv2(x))
        x = self.pooling2(x)
        x = x.view(1,-1)
        x = self.active_fun(self.fully_connect1(x))
        x = self.active_fun(self.fully_connect2(x))
        x = self.output(x)
        #squeeze
        #x = torch.squeeze(x)
        return x





    
