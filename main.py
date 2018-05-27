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


import numpy as np
from data.dataloader import HDF5PatchesDataset

filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50
training = 1
input_size = 8192


class AudioWonderNet(nn.Module):
    def __init__(self, blocks):
        super(AudioWonderNet, self).__init__()

        self.features = nn.Sequential()

        self.count = 0

        conv_input = 1
        output = 16
        fc_in = input_size//output # compute fc size pls

        for b in range(0,blocks):
            i = b+1
            self.features.add_module("conv"+str(i),nn.Conv1d(conv_input, output, filter_size, stride=1, padding=1)), # padding/stride?
            self.features.add_module("bn"+str(i),nn.BatchNorm1d(output)),
            self.features.add_module("relu"+str(i),nn.LeakyReLU()),
            self.features.add_module("pool"+str(i),nn.MaxPool1d(2))
            conv_input = output
            output = conv_input * 2

        print(self.features)
        self.final = nn.Linear(557056, num_classes) # after features block we have a tensor of 1, 557056


    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1) # reshapes tensor, replacing fc layer - dumdum
        # print('yo h:', h.shape)
        h = self.final(h)
        return h

net = AudioWonderNet(4)
# optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
# loss_function = nn.CrossEntropyLoss()

# # GPU STUFF
# dtype = torch.FloatTensor
# num_gpus = torch.cuda.device_count()
# loss_log = []

# # Check how many GPUs, do cuda/DataParallel accordingly
# if num_gpus > 0:
#     dtype = torch.cuda.FloatTensor
#     net.cuda()
#     loss_function.type(dtype)

# if num_gpus > 1:
#     net = nn.DataParallel(net).cuda()

# / GPU STUFF


# CHECK POINT AND PLOT STUFF
checkpoint_label = 'raw'
checkpoint_epoch = 0
checkpoint_dir = 'checkpoints/'

tensor_label = 'tb_raw_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
tensorboard = False
plot = 0




# DUMMY DATA
# if not training:
#     test_in = Variable(torch.from_numpy(np.sin(np.linspace(0, 2*np.pi, 8192)))).unsqueeze(0).unsqueeze(0).float()
#     print('input shape: ', test_in.shape)
#     outties = net(test_in)
#     print('output shape: ', outties.shape)



# REAL DATA
train_dataset = HDF5PatchesDataset('data/train_pesq.hdf5')
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)

# Try and load the checkpoint
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("\nCreated a 'checkpoints' folder to save/load the model")

try:
    # load the checkpoint
    filename = '{:s}checkpoint_{:s}_epoch_{:06d}.pt'.format(checkpoint_dir, checkpoint_label, checkpoint_epoch)
    checkpoint = torch.load(filename)

    # set the model state
    net.load_state_dict(checkpoint['state_dict'])

    # set optimizer state
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    starting_epoch = checkpoint['epoch']
    loss_log = checkpoint['loss_log']

    print("\nLoaded checkpoint: " + filename)
except FileNotFoundError:
    print("\nNo checkpoint found, starting training")


# TensorboardX init
if not os.path.exists("logs/"+ tensor_label):
    os.makedirs("logs/" + tensor_label)

writer = SummaryWriter('./logs/' + tensor_label)

# training

if(training):
    for epoch in range(starting_epoch, num_epochs):

        for i, (x, y) in enumerate(train_dataloader):

            # x_var = Variable(x.type(dtype))
            # y_var = Variable(y.type(dtype))
            x_var = x.cuda(non_blocking=True)
            y_var = y.cuda(non_blocking=True).type(torch.cuda.LongTensor)


            # Forward pass
            out = net(x_var)
            # Compute loss
            print(y_var.data)
            loss = loss_function(out, y_var)
            loss_log.append(loss.item())
            # Zero gradients before the backward pass
            optimizer.zero_grad()
            # Backprop
            loss.backward()
            # Update the params
            optimizer.step()

            print(loss.item())
            if tensorboard:
                writer.add_scalar('train/loss', loss.item(), plot)
                plot += 1

    # Save checkpoint
    if (epoch % checkpoint_every_epochs == 0 or epoch == (num_epochs-1)) and (epoch != starting_epoch):
        save_file = '{:s}checkpoint_{:s}_epoch_{:06d}.pt'.format(checkpoint_dir, checkpoint_label, epoch)
        save_state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_loss' : scheduler.best,
            'bad_epoch' : scheduler.num_bad_epochs,
            'loss_log' : loss_log
        }
        torch.save(save_state, save_file)
        print('\nCheckpoint saved')


            

