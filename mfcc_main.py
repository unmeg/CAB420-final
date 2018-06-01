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

from net_evaluator import NetEvaluator


import numpy as np
from data.dataloader import HDF5PatchesDataset


# Parameters
checkpoint_label = 'mfcc'
filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50 # this gives us a category for every step of 0.1 in the PESQ score (0-5)
training = 1 # Boolean setting for training mode

class AudioMagicNet(nn.Module):
    def __init__(self, blocks):
        super(AudioMagicNet, self).__init__()

        self.features = nn.Sequential() # a module that acts quite like a list/array

        # Starting values
        conv_input = 1
        output = 16 
        
        for b in range(0,blocks):
            i = b+1
            self.features.add_module("conv"+str(i),nn.Conv2d(conv_input, output, filter_size, stride=1, padding=1)), # padding maintains size
            self.features.add_module("bn"+str(i),nn.BatchNorm2d(output)),
            self.features.add_module("relu"+str(i),nn.LeakyReLU()),
            self.features.add_module("pool"+str(i),nn.MaxPool2d(2))
            conv_input = output
            output = conv_input * 2

        print(self.features)
        self.final = nn.Linear(1920, num_classes) # after features block we have a tensor of [1, 1920]

    def forward(self, x):
        h = self.features(x)
        # print('h shape: ', h.shape)
        h = h.view(h.size(0), -1)  # reshapes tensor into [1, x]
        h = self.final(h)
        return h

net = AudioMagicNet(4) # Instantiates the network with 4 layers

dataset = HDF5PatchesDataset('../dataset/train_pesq_small.hdf5')

if training:
    evaluator = NetEvaluator(net, dataset, mfcc=True, checkpoint_label=checkpoint_label)
    evaluator.eval()
