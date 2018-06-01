# The block set-up below is modeled on this PyTorch example: https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import os
import time
import datetime

from torch.utils.data import DataLoader

from testies import Testies


import numpy as np
from data.dataloader import HDF5PatchesDataset

# Parameters
filter_size = 65
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50 # this gives us a category for every step of 0.1 in the PESQ score (0-5)
training = 1 # Boolean setting for training mode
input_size = 8192


class AudioWonderNet(nn.Module):
    def __init__(self, blocks):
        super(AudioWonderNet, self).__init__()

        self.features = nn.Sequential() # a module that acts quite like a list/array

        self.count = 0

        # Starting values
        conv_input = 1
        output = 16

        # Creates the specified number of feature extraction layers
        for b in range(0,blocks):
            i = b+1
            self.features.add_module("conv"+str(i),nn.Conv1d(conv_input, output, filter_size, stride=1, padding=32)), # padding maintains size
            self.features.add_module("bn"+str(i),nn.BatchNorm1d(output)),
            self.features.add_module("relu"+str(i),nn.LeakyReLU()),
            self.features.add_module("pool"+str(i),nn.MaxPool1d(2))
            conv_input = output
            output = conv_input * 2

        print(self.features)
        self.final = nn.Linear(65536, num_classes) # after features block we have a tensor of 1, 65536


    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1) # reshapes tensor into [1, x]
        # print('yo h:', h.shape)
        h = self.final(h)
        return h

net = AudioWonderNet(4) # Instantiates the network with 4 layers

dataset = HDF5PatchesDataset('/home/mining-test/dataset/train_pesq_large.hdf5')

if training:
    testies = Testies(net, dataset)
    testies.eval()




