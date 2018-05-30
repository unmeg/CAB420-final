import os
import h5py
import soundfile as sf
from pesq_score import *
#import sounddevice as sd
import time
import random
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate
import librosa
import matplotlib.pyplot as plt
from multiprocessing import Pool


write_every_patches = 10
max_patches = 100
num_classes = 50
pesq_sr = 16000
input_hdf5 = 'train_vctk_patches.hdf5'
output_hdf5 = 'train_pesq_large.hdf5'


def score2index(score):
    return round(score, 1) * (num_classes / 5)


def index2score(index):
    return (i / (num_classes / 5))


def encode_score(score):
    i = score2index(score)
    z = np.zeros(num_classes)
    if i != 0:
        z[int(i)] = 1
    return z # np.expand_dims(z, axis=0)


def decode_score(score):
    i = np.where(score==1)[0]
    return index2score(i)


def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp


def resample(x, original_sr, new_sr, filename, noise):
    r = original_sr // new_sr
    print('\n', x.shape)
    print(r)
    resampled = x[::r].copy() # librosa.resample(x.copy(), original_sr, new_sr, res_type='kaiser_fast')
    print('sub', resampled.shape)
    resampled = upsample(resampled, r)
    print('up', resampled.shape)

    # generate cracking noise thing
    if noise:
        r = np.random.rand(resampled.shape[0]) * 0.5
        b = r < 0.001
        r[~b] = 0
        r[b] = np.random.rand(1) * 0.05
        resampled += r
    filename_out = os.path.join('temp', str(new_sr) + '_' + str(noise) + '_' + filename)
    sf.write(filename_out, resampled, pesq_sr, 'PCM_16')
    print('out', resampled.shape)
    return (filename_out, resampled)

# get patches dataset into memory
input_hdf5_file = h5py.File(input_hdf5, 'r')
hr_dataset = input_hdf5_file['hr']
print(hr_dataset.shape)

# open output hdf5 file and prepare for resizing
output_hdf5_file = h5py.File(output_hdf5, 'w')
x_dataset = output_hdf5_file.create_dataset("x", (1, 1, 8192), maxshape=(None, 1, 8192), dtype=np.float32)
y_dataset = output_hdf5_file.create_dataset("y", (1,), maxshape=(None,), dtype=np.int32)

ni = 0
n = min(max_patches, hr_dataset.shape[0])
indexes = list(range(n))
random.shuffle(indexes)
print('patches:', n)
print('input shape:', hr_dataset.shape)

scores = []
patches = []

for hi in indexes:

    ni += 1
    print('{:.2f}%'.format(ni * 100 / n))

    # grab a hr patch from the hdf5
    x = hr_dataset[hi, 0, :]
    hr_filename = '{:d}_hr.wav'.format(hi)

    # store filenames in here of the downsampled and noisy (degraded) versions of this patch
    these_scores = []
    degraded_x = []

    # take the hr reference and downsample it to the following sr's, make one with noise and one without for each
    sampling_rates = [16000, 8000, 4000, 2000, 1000]
    for sr in sampling_rates:
        degraded_x.append(resample(x, 16000, sr, hr_filename, False))
        degraded_x.append(resample(x, 16000, sr, hr_filename, True))

    # get the score of every degraded version
    for i in range(len(degraded_x)):
        try:
            score = get_pesq(degraded_x[0][0], degraded_x[i][0])
            these_scores.append(score2index(score))
            patches.append(np.append(degraded_x[0][1], degraded_x[-1][1]))
        except Exception as e:
            pass
        finally:
            if i > 0:
                pass
                os.remove(degraded_x[i][0])

    os.remove(degraded_x[0][0])

    # print('\n')
    # for di in range(len(these_scores)):
    #     print('{:.3f}\t{:s}'.format(these_scores[di], os.path.basename(degraded_x[di][0])))

    scores += these_scores

    if len(patches) >= write_every_patches:
        patches = np.expand_dims(np.array(patches, dtype=np.float32), axis=1)
        scores = np.array(scores, dtype=np.int32)
        print(patches.shape, scores.shape)
        assert patches.shape[0] == scores.shape[0], "patches and score size don't match!"
        x_dataset.resize(x_dataset.shape[0] - 1 + patches.shape[0], axis=0)
        y_dataset.resize(y_dataset.shape[0] - 1 + scores.shape[0], axis=0)
        print(x_dataset.shape, y_dataset.shape)
        x_dataset[x_dataset.shape[0] - write_every_patches : x_dataset.shape[0], :, :] = patches
        y_dataset[y_dataset.shape[0] - write_every_patches : y_dataset.shape[0]] = scores
        patches = []
        scores = []

# a = np.array(hr_dataset)[1:10]
# print(a.shape)

# patches = []
# scores = []

# with Pool(8) as p:
#     res = p.imap_unordered(generate, a)
#     for x in res:
#         if len(x[0]) > 0 and len(x[1]) > 0:
#             for i in range(len(x[0][0])):
#                 patches.append(x[0][i])
#                 scores.append(x[])

# patches = np.expand_dims(np.array(patches, dtype=np.float32), axis=1)
# scores = np.array(scores, dtype=np.int32)

# print(patches.shape)
# print(scores.shape)

# with h5py.File(output_hdf5, 'w') as f:
#     x = f.create_dataset("x", patches.shape, dtype=np.float32)
#     y = f.create_dataset("y", scores.shape, dtype=np.int32)

#     x[...] = patches
#     y[...] = scores

h5_file = h5py.File(output_hdf5, 'r')
x = h5_file['x']
y = h5_file['y']
print(x.shape, y.shape)

