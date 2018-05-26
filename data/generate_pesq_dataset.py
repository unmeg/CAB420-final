import os
import h5py
import soundfile as sf
from pesq_score import *
import sounddevice as sd
import time
import random
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate
import librosa


pesq_sr = 16000
input_hdf5 = 'train_vctk_patches.hdf5'
output_hdf5 = 'train_pesq.hdf5'


def upsample(x_lr, d, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f, der=d)

    return x_sp


def resample(x, original_sr, new_sr, filename, noise):

    x = librosa.resample(x, original_sr, new_sr, res_type='kaiser_fast')

    # generate cracking noise thing
    if noise:
        r = np.random.rand(x.shape[0]) * 0.5
        b = r < 0.001
        r[~b] = 0
        r[b] = np.random.rand(1) * 0.05
        x += r

    filename_out = os.path.join('temp', str(sr) + '_' + str(noise) + '_' + filename)
    sf.write(filename_out, x, pesq_sr, 'PCM_16')
    return filename_out


h5_file = h5py.File('train_patches.hdf5', 'r')
hr_dataset = h5_file['hr']

scores = []
patches = []

indexes = random.shuffle(range(hr_dataset.shape[0]))
for i in indexes:

    # grab a hr patch from the hdf5
    x = hr_dataset[i, 0, :]
    hr_filename = '{:d}_hr.wav'.format(i)

    # store filenames in here of the downsampled and noisy (degraded) versions of this patch
    degraded_files = []

    # take the hr reference and downsample it to the following sr's, make one with noise and one without for each
    sampling_rates = [16000, 8000, 4000, 2000, 1000]
    for sr in sampling_rates:
        degraded_files.append('temp', resample(x, pesq_sr, sr, hr_filename, False))
        degraded_files.append('temp', resample(x, pesq_sr, sr, hr_filename, True))

    for i in range(len(degraded_files)):
        try:
            score = get_pesq(degraded_files[0], degraded_files[i])
            scores.append(score)
            patches.append(np.append(x, y))
        except:
            pass
        finally:
            if i > 0:
                os.remove(degraded_files[i])

    os.remove(degraded_files[0])

    for i in range(len(degraded_files)):
        print('{:.3f}\t{:s}'.format(scores[i], os.path.basename(degraded_files[i])))

patches = np.array(patches)
scores = np.array(scores)

with h5py.File('pesq_test.hdf5', 'w') as f:
    x = f.create_dataset("x", patches.shape, dtype=np.float32)
    y = f.create_dataset("y", scores.shape, dtype=np.float32)

    x[...] = patches
    y[...] = scores

h5_file = h5py.File('pesq_test.hdf5', 'r')
x = h5_file['x']
y = h5_file['y']

print(x.shape, y.shape)
