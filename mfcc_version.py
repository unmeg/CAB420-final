
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display


n_mfcc = 13
n_mels = 40
n_fft = 512
win_length = 400 # 0.025*16000
hop_length = 160 # 0.010 * 16000
window = 'hamming'
fmin = 20
fmax = 4000
filter_size = 3
learning_rate = 1e-4
starting_epoch = 0
num_epochs = 50
num_classes = 50 # gives us a category for every half step?
training = 0
input_size = 8192


## USING LIBROSA
MELWINDOW = 200
MELBANK = 80

y, sr = librosa.load('test.wav')

s = np.abs(librosa.core.stft(y=y, n_fft=800, hop_length=MELWINDOW, window='hann', center=True)) # pre-computed power spec
test_input = librosa.feature.melspectrogram(S=s, n_mels=MELBANK, fmax=7600, fmin=125, power=2, n_fft = 800, hop_length=MELWINDOW) # passed to melfilters
print('input shape1', test_input.shape)
test_input = librosa.core.amplitude_to_db(S=test_input, ref=1.0, amin=5e-4, top_db=80.0) #logamplitude
test_input = Variable(torch.from_numpy(test_input).float()).unsqueeze(0).unsqueeze(0)
# librosa.display.specshow(voice_input_oh, y_axis='log', x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

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
        self.final = nn.Linear(26880, num_classes) # hardcoded based on known size (h.shape) >>> 128 x 5 x 42
        
        
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

# dummy data


print('input shape1', test_input.shape) # 80, 678
outties = net(test_input)
print(outties.shape)


# # training 

if(training):
    for epoch in range(starting_epoch, num_epochs):
    
        for i, (x, y) in enumerate(train_dataloader):
            
            # x_var = Variable(x.type(dtype))
            # y_var = Variable(y.type(dtype))
            x_var = x.cuda(non_blocking=True)
            y_var = y.cuda(non_blocking=True)
           
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



###### CODE CEMETARY #######


# def wav_to_mfcc(file):
#     # https://github.com/jameslyons/python_speech_features/blob/master/example.py
#     # signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=<function <lambda>>
#     fs, signal = wav.read(file)
#     mfcc_feat = mfcc(signal,fs, numcep=80)
#     #d_mfcc_feat = delta(mfcc_feat, 2) # change
#     fbank_feat = logfbank(signal,fs, nfilt=80) # change)

#     # return mfcc_feat
#     return fbank_feat

# thing1 = wav_to_mfcc('test.wav')
# thing2 = wav_to_mfcc('test-y.wav')
# plt.plot(thing1)
# plt.show()
# print(thing1.shape)
# print(thing2.shape)

# temp =  thing1[ : , np.newaxis , :] # adds 1 in the middle, not sure why i'd want that tbh
# temp2=  thing2[ : , np.newaxis , :]

# #print(temp2.shape)
# input_var = Variable(torch.Tensor(temp))
# input_var2 = Variable(torch.Tensor(temp2))

# print(input_var.shape)
# print(input_var2.shape)

# y, sr = librosa.load('test.wav')
# # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=100, fmax = 8000, n_mels=QUANT)
# melspec = librosa.feature.melspectrogram(y=y, sr=sr)
#

# print(melspec)

# print(melspec.shape) #128, 265

# melspec = melspec[:,:-1]

# print(melspec)

# print(melspec.shape)#128, 264

# plt.plot(melspec)
# plt.show()


#n_fff is window size

# s = np.abs(librosa.core.stft(y=y, n_fft=800, hop_length=MELWINDOW, window='hann', center=True)) # pre-computed power spec
# voice_input_oh = librosa.feature.melspectrogram(S=s, n_mels=MELBANK, fmax=7600, fmin=125, power=2, n_fft = 800, hop_length=MELWINDOW) # passed to melfilters
# voice_input_oh = librosa.core.amplitude_to_db(S=voice_input_oh, ref=1.0, amin=5e-4, top_db=80.0) #logamplitude
# librosa.display.specshow(voice_input_oh, y_axis='log', x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()


# print(voice_input_oh)
# print(voice_input_oh.shape) # 80 coefficients, 678 frames

# plt.plot(voice_input_oh)
# plt.show()

# melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MELBANK, fmax=7600, fmin=125, power=2, n_fft = 800, hop_length=MELWINDOW)
# # melspec = librosa.feature.melspectrogram(y=y, sr=sr)
# # melspec = melspec[:,:-1]
# print(melspec.shape)
# librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()

# plt.figure()

# D = np.abs(librosa.stft(y))**2
# S = librosa.feature.melspectrogram(S=D, n_mels=MELBANK, fmax=7600, fmin=125, power=2, n_fft = 800, hop_length=MELWINDOW)
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# librosa.display.specshow(librosa.power_to_db(melspec, ref=1.0), y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram2')
# plt.tight_layout()
# plt.show()

# plt.figure()