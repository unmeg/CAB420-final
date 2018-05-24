# batch norm paper: https://arxiv.org/abs/1502.03167
# example: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn16s.py
# block setup borrowed from: https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_classes = 5

class AudioWonderNet(nn.Module):
    def __init__(self):
        super(AudioWonderNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, filter_size, stride=1, padding=1), # padding/stride?
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, filter_size, 1, 1), # padding?
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, filter_size, 1, 1), # padding?
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, filter_size, 1, 1), # padding?
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )

        self.final = nn.Linear(256, 128) #nfi what these vals should be

    def forward(self, x):
        x = self.block1(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.block3(x)
        print(x.shape)
        x = self.block4(x)
        print(x.shape)
        x = self.final(x)
        print(x.shape)

        return x

net = AudioWonderNet()
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

test_in = Variable(torch.from_numpy(np.random.randn(1,1,4096))).float()
print(test_in)
outties = net(test_in)

# # training 

# for epoch in range(starting_epoch, num_epochs):
   
#     for i, (x, y) in enumerate(train_dataloader):

#         x_var = Variable(x.type(dtype))
#         y_var = Variable(y.type(dtype))

#         out = net(x_var)

#         optimizer.zero_grad()
#         loss = loss_function(out, y_var)
#         loss.backward()
#         optimizer.step()

#         loss_log.append(loss.item())

