# batch norm paper: https://arxiv.org/abs/1502.03167
# example: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn16s.py
# block setup borrowed from: https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

filter_size = 3

class AudioWonderNet(nn.Module):
    def __init__(self):
        super(AudioWonderNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, filter_size, 1, 1), # padding/stride?
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

        self.final = nn.Linear(VALS)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)

        return x

net = AudioWonderNet()
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()