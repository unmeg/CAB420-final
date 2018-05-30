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
from data.dataloader import HDF5PatchesDataset

import numpy as np

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display


n_mels = 80
n_fft = 512
win_length = 400 # 0.025 x 16000
hop_length = 160 # 0.010 x 16000
window = 'hamming'
fmin = 20
fmax = 4000
filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50 # gives us a category for every half step?
training = 1
input_size = 8192
batch_size=64



# y, sr = librosa.load('test.wav')

# s = np.abs(librosa.core.stft(y=y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)) # pre-computed power spec
# test_input = librosa.feature.melspectrogram(S=s, n_mels=n_mels, fmax=7600, fmin=125, power=2, n_fft = n_fft, hop_length=hop_length) # passed to melfilters == hop_length used to be MELWINDOW
# print('input shape1', test_input.shape)
# test_input = librosa.core.amplitude_to_db(S=test_input, ref=1.0, amin=5e-4, top_db=80.0) #logamplitude
# test_input = Variable(torch.from_numpy(test_input).float()).unsqueeze(0).unsqueeze(0)
# librosa.display.specshow(test_input, y_axis='log', x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

class AudioMagicNet(nn.Module):
    def __init__(self, blocks):
        super(AudioMagicNet, self).__init__()

        self.features = nn.Sequential()

        conv_input = 1
        output = 16 # get the size
        fc_in = input_last_out = 0

        for b in range(0,blocks):
            i = b+1
            self.features.add_module("conv"+str(i),nn.Conv2d(conv_input, output, filter_size, stride=1, padding=1)), # padding/stride?
            self.features.add_module("bn"+str(i),nn.BatchNorm2d(output)),
            self.features.add_module("relu"+str(i),nn.LeakyReLU()),
            self.features.add_module("pool"+str(i),nn.MaxPool2d(2))
            conv_input = output
            output = conv_input * 2

        print(self.features)
        self.final = nn.Linear(17280 * batch_size, num_classes) # hardcoded based on known size (h.shape) >>> 128 x 5 x 42

    def forward(self, x):
        h = self.features(x)
        print('h shape: ', h.shape)
        h = h.view(h.size(0), -1)
        h = self.final(h)
        return h

net = AudioMagicNet(4)
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

## GPU STUFF
dtype = torch.FloatTensor
num_gpus = torch.cuda.device_count()
loss_log = []

# Check how many GPUs, do cuda/DataParallel accordingly
if num_gpus > 0:
    dtype = torch.cuda.FloatTensor
    net.cuda()

if num_gpus > 1:
    net = nn.DataParallel(net).cuda()

## / GPU

# CHECK POINT AND PLOT STUFF
checkpoint_label = 'mfcc'
checkpoint_epoch = 0
checkpoint_dir = 'checkpoints/'

tensor_label = 'tb_mfcc_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
tensorboard = False
plot = 0


# dummy data


#print('input shape1', test_input.shape) # 80, 678
#outties = net(test_input)
#print(outties.shape)


train_dataset = HDF5PatchesDataset('data/train_pesq.hdf5')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 0, shuffle=True)

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

# # training 

if(training):
    for epoch in range(starting_epoch, num_epochs):
    
        for i, (x, y) in enumerate(train_dataloader):
            
            # x_var = Variable(x.type(dtype))
            # y_var = Variable(y.type(dtype))
      #      x_var = x.cuda(non_blocking=True)
            y_var = y.cuda(non_blocking=True).type(torch.cuda.LongTensor)
            x_vals = []
            
            for thing in range(0, batch_size):
                go_in = x[thing,0,:]
                print('in size: ', go_in.shape)
                s = np.abs(librosa.core.stft(y=x[thing,0,:].numpy(), n_fft=n_fft, hop_length=hop_length, window='hann', center=True)) # pre-computed power spec
                test_input = librosa.feature.melspectrogram(S=s, n_mels=n_mels, fmax=7600, fmin=125, power=2, n_fft = n_fft, hop_length=hop_length) # passed to melfilters == hop_length used to be 200
                print('melspec size: ', test_input.shape)
                x_vals.append(librosa.core.amplitude_to_db(S=test_input, ref=1.0, amin=5e-4, top_db=80.0)) #logamplitude)
                
            x_hold = np.concatenate( x_vals, axis=0 )
            print('full batch size: ', x_hold.shape)

            x_var = Variable(torch.from_numpy(x_hold).float()).unsqueeze(0).unsqueeze(0)
            
            x_var = x_var.cuda(non_blocking=True)

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
