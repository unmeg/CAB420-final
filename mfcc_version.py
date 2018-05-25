
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

num_features = 13

def wav_to_mfcc(file):
    # https://github.com/jameslyons/python_speech_features/blob/master/example.py
    fs, signal = wav.read(file)
    mfcc_feat = mfcc(signal,fs)
    #d_mfcc_feat = delta(mfcc_feat, 2) # change
    #fbank_feat = logfbank(sig,rate) # change

    return mfcc_feat

thing1 = wav_to_mfcc('test.wav')
thing2 = wav_to_mfcc('test-y.wav')
plt.plot(thing1)
plt.show()
print(thing1.shape)
print(thing2.shape)

temp =  thing1[ : , np.newaxis , :] # adds 1 in the middle, not sure why i'd want that tbh
temp2=  thing2[ : , np.newaxis , :]

#print(temp2.shape)
input_var = Variable(torch.Tensor(temp))
input_var2 = Variable(torch.Tensor(temp2))

print(input_var.shape)
print(input_var2.shape)


## USING LIBROSA
MELWINDOW = 200
MELBANK = 80

y, sr = librosa.load('test.wav')
# melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=100, fmax = 8000, n_mels=QUANT)
melspec = librosa.feature.melspectrogram(y=y, sr=sr)
print(melspec)

print(melspec.shape) #128, 265

melspec = melspec[:,:-1]

print(melspec)

print(melspec.shape)#128, 264

plt.plot(melspec)
plt.show()


s = np.abs(librosa.core.stft(y=y, n_fft=800, hop_length=MELWINDOW, window='hann', center=True))
voice_input_oh = librosa.feature.melspectrogram(S=s, n_mels=MELBANK, fmax=7600, fmin=125, power=2, n_fft = 800, hop_length=MELWINDOW)
voice_input_oh = librosa.core.amplitude_to_db(S=voice_input_oh, ref=1.0, amin=5e-4, top_db=80.0) #logamplitude
librosa.display.specshow(voice_input_oh)
print(voice_input_oh)
print(voice_input_oh.shape) # 80, 678

plt.plot(voice_input_oh)
plt.show()



# class AudioMagicNet(nn.Module):
#     def __init__(self, blocks):
#         super(AudioWonderNet, self).__init__()

#         self.features = nn.Sequential()

#         conv_input = 1
#         output = 16 # get the size
#         fc_in = input_size//output # compute fc size pls

#         for b in range(0,blocks):
#             i = b+1
#             self.features.add_module("conv"+str(i),nn.Conv1d(conv_input, output, filter_size, stride=1, padding=1)), # padding/stride?
#             self.features.add_module("bn"+str(i),nn.BatchNorm1d(output)),
#             self.features.add_module("relu"+str(i),nn.LeakyReLU()),
#             self.features.add_module("pool"+str(i),nn.MaxPool1d(2))
#             conv_input = output
#             print()
#             output = conv_input * 2

#         print(self.features)

#         self.final1 = nn.Linear(fc_in, conv_input)
#         self.final2 = nn.Linear(128, num_classes)
        
        
#     def forward(self, x):
#         h = self.features(x)
#         h = self.final1(h)
#         print(h.shape)
#         h = self.final2(h)
#         print(h.shape)
#         return h