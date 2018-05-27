# batch norm paper: https://arxiv.org/abs/1502.03167
# example: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn16s.py
# block setup borrowed from: https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py

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

filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50
training = 1

input_size = 678

class AudioMagicNet(nn.Module):
    def __init__(self, blocks):
        super(AudioMagicNet, self).__init__()

        self.features = nn.Sequential()

        conv_input = 1
        output = 16 # get the size
        fc_in = input_size//output # compute fc size pls

        for b in range(0,blocks):
            i = b+1
            self.features.add_module("conv"+str(i),nn.Conv2d(conv_input, output, filter_size, stride=1, padding=1)), # padding/stride?
            self.features.add_module("bn"+str(i),nn.BatchNorm2d(output)),
            self.features.add_module("relu"+str(i),nn.LeakyReLU()),
            self.features.add_module("pool"+str(i),nn.MaxPool2d(2))
            conv_input = output
            output = conv_input * 2

        print(self.features)
        self.final = nn.Linear(17280, num_classes) # hardcoded based on known size (h.shape) >>> 128 x 5 x 42

    def forward(self, x):
        h = self.features(x)
        # print('h shape: ', h.shape)
        h = h.view(h.size(0), -1)
        h = self.final(h)
        return h

net = AudioMagicNet(4)

dataset = HDF5PatchesDataset('data/train_pesq.hdf5')

if training:
    testies = Testies(net, dataset, mfcc=True, checkpoint_label='mfcc')
    testies.eval()
