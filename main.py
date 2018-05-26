# batch norm paper: https://arxiv.org/abs/1502.03167
# example: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn16s.py
# block setup borrowed from: https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50
training = 0
input_size = 8192

class AudioWonderNet(nn.Module):
    def __init__(self, blocks):
        super(AudioWonderNet, self).__init__()

        self.features = nn.Sequential()

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
        self.final = nn.Linear(128 * 512, num_classes) # after features block we have a tensor of [1, 128, 512]. 
        
        
    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1) # reshapes tensor, replacing fc layer - dumdum
        print('yo h:', h.shape)
        h = self.final(h)
        return h

net = AudioWonderNet(4)
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

test_in = Variable(torch.from_numpy(np.sin(np.linspace(0, 2*np.pi, 8192)))).unsqueeze(0).unsqueeze(0).float()
print('input shape: ', test_in.shape)
outties = net(test_in)
print('output shape: ', outties.shape)

# # training 

if(training):
    for epoch in range(starting_epoch, num_epochs):
    
        for i, (x, y) in enumerate(train_dataloader):

            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype))
            
            # Forward pass
            out = net(x_var)
            # Compute loss
            loss = loss_function(out, y_var)       
            loss_log.append(loss.item())
            # Zero gradients before the backward pass
            optimizer.zero_grad()
            # Backprop
            loss.backward()
            # Update the params
            optimizer.step()

