# batch norm paper: https://arxiv.org/abs/1502.03167

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
            nn.Conv1d(1, 16, filter_size), # padding?
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(HOW BIG?)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, filter_size), # padding?
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(HOW BIG?)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, filter_size), # padding?
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(HOW BIG?)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, filter_size), # padding?
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(HOW BIG?)
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