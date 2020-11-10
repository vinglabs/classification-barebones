import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class TwoLayerConv100RGB(nn.Module):
    def __init__(self):
        super(TwoLayerConv100RGB,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*22*22,10)
        self.fc2 = nn.Linear(10,2)


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = TwoLayerConv100RGB()
# a = model(torch.tensor(np.random.rand(1,3,100,100),dtype=torch.float32))
# print(a.shape)
#


