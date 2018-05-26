import os
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate
import h5py
import librosa
import soundfile as sf


def get_speakers(directory):
    speaker_dirs = []
    for dirname in sorted(os.listdir(directory)):
        dirname = os.path.join(directory, dirname)
        if os.path.isdir(dirname):
            speaker_dirs.append(dirname)
    return speaker_dirs


def get_wav_files(directory):
    files = []
    for filename in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, filename)) and filename.lower().endswith('.wav'):
            files.append(filename)
    return files


if __name__ == '__main__':

    # directories
    vctk_dir = '/home/lindsay/Documents/VCTK testing/'
    train_hdf5_filename = 'train_vctk_patches.hdf5'
    val_hdf5_filename = 'val_vctk_patches.hdf5'
    hr = 'HR'
    lr = 'LR'

    # dataset settings
    single_speaker = True
    train_val_ratio = 0.90
    percent_dataset = 0.05

    # hr/lr sample rate settings
    vctk_sample_rate = 48000
    hr_sample_rate = 16000
    lr_sample_rate = 8000

    # patching settings
    dimension = 4096
    stride = 2048
    batch_size = 1

    # auto vars
    r = hr_sample_rate // lr_sample_rate
    train_hr_patches = []
    train_lr_patches = []
    val_hr_patches = []
    val_lr_patches = []

    # get directories for each speaker
    speaker_dirs = get_speakers(vctk_dir)

    if single_speaker:
        speaker_dirs = speaker_dirs[0:1]

    ls = len(speaker_dirs)
    speaker_dirs = speaker_dirs[:math.ceil(percent_dataset * ls)]
    ls = len(speaker_dirs)

    for si, speaker in enumerate(speaker_dirs):

        print('\nSpeaker {:d} of {:d} - {:.0f}%'.format(si+1, ls, (si+1)/ls * 100))

        # create HR and LR directories
        # hr_dir = os.path.join(speaker, hr)
        # lr_dir = os.path.join(speaker, lr)
        # if not os.path.isdir(hr_dir):
        #     os.mkdir(os.path.join(speaker, hr))
        # if not os.path.isdir(lr_dir):
        #     os.mkdir(os.path.join(speaker, lr))

        # iterate over every wav file for this speaker
        wavs = get_wav_files(speaker)
        lw = len(wavs)
        for wi, wav in enumerate(wavs):

            # load and resample the orignal wav to get high res version
            hr_wav, fs = librosa.load(os.path.join(speaker, wav), sr=hr_sample_rate, res_type='kaiser_fast')

            # downsample the hr version to get the lr version
            # lr_wav = librosa.core.resample(hr_wav, hr_sample_rate, lr_sample_rate, res_type='kaiser_fast')
            # lr_wav = hr_wav[::r].copy()

            # then upsample using interpolation to get the actual input for the network
            # input_wav = upsample(lr_wav, r)

            # input_wav = upsample(lr_wav, 0, r)

            hr_wav = librosa.util.normalize(hr_wav, axis=None)
            # lr_wav = librosa.util.normalize(lr_wav, axis=None)
            # input_wav = librosa.util.normalize(input_wav, axis=None)

            # print(lr_sample_rate, lr_wav.shape, np.min(lr_wav), np.max(lr_wav))
            # print(hr_sample_rate, hr_wav.shape, np.min(hr_wav), np.max(hr_wav))
            # print(hr_sample_rate, input_wav.shape, np.min(input_wav), np.max(input_wav))

            # save the lr and hr wav files
            # sf.write(os.path.join(hr_dir, 'hr_' + wav), hr_wav, hr_sample_rate, 'PCM_16')
            # sf.write(os.path.join(lr_dir, 'lr_' + wav), lr_wav, lr_sample_rate, 'PCM_16')

            # generate patches for every wav file
            # max_i = min(len(hr_wav), len(input_wav)) - dimension + 1
            max_i = len(hr_wav) - dimension + 1

            for i in range(0, max_i, stride):

                hr_patch = np.array(hr_wav[i : i + dimension]).reshape((1, dimension))
                # lr_patch = np.array(input_wav[i : i + dimension]).reshape((1, dimension))

                # if we're in single speaker mode, we use the last (1-ratio) of wav's for validation
                # if we're not, we use the last (1-ratio) of speaker's for validation
                if (wi <= np.floor(lw * train_val_ratio) and single_speaker) or \
                   (si <= np.floor(ls * train_val_ratio) and not single_speaker):
                    train_hr_patches.append(hr_patch)
                    # train_lr_patches.append(lr_patch)
                else:
                    val_hr_patches.append(hr_patch)
                    # val_lr_patches.append(lr_patch)

            if wi % (lw // 4) == 0:
                print('wav {:d} of {:d} - {:.0f}%'.format(wi+1, lw, (wi+1)/lw * 100))


    # trim patches to multiple of batch_size
    num_patches = len(train_hr_patches)
    num_to_keep = int(np.floor(num_patches / batch_size) * batch_size)
    train_hr_patches = np.array(train_hr_patches[:num_to_keep], dtype=np.float32)
    # train_lr_patches = np.array(train_lr_patches[:num_to_keep], dtype=np.float32)

    num_patches = len(val_hr_patches)
    num_to_keep = int(np.floor(num_patches / batch_size) * batch_size)
    val_hr_patches = np.array(val_hr_patches[:num_to_keep], dtype=np.float32)
    # val_lr_patches = np.array(val_lr_patches[:num_to_keep], dtype=np.float32)

    print('\ntrain HR patches shape:', train_hr_patches.shape)
    # print('train LR patches shape:', train_lr_patches.shape)

    print('\ntrain HR patches min, max, mean:', np.min(train_hr_patches), np.max(train_hr_patches), np.mean(train_hr_patches))
    # print('train LR patches min, max, mean:', np.min(train_lr_patches), np.max(train_lr_patches), np.mean(train_lr_patches))

    print('\nval HR patches shape:', val_hr_patches.shape)
    # print('val LR patches shape:', val_lr_patches.shape)

    print('\nval HR patches min, max, mean:', np.min(val_hr_patches), np.max(val_hr_patches), np.mean(val_hr_patches))
    # print('val LR patches min, max, mean:', np.min(val_lr_patches), np.max(val_lr_patches), np.mean(val_lr_patches))

    # save patches in hdf5 files
    with h5py.File(train_hdf5_filename, 'w') as f:
        hr_dataset = f.create_dataset("hr", train_hr_patches.shape, dtype=np.float32)
        # lr_dataset = f.create_dataset("lr", train_lr_patches.shape, dtype=np.float32)

        hr_dataset[...] = train_hr_patches
        # lr_dataset[...] = train_lr_patches

    with h5py.File(val_hdf5_filename, 'w') as f:
        hr_dataset = f.create_dataset("hr", val_hr_patches.shape, dtype=np.float32)
        # lr_dataset = f.create_dataset("lr", val_lr_patches.shape, dtype=np.float32)

        hr_dataset[...] = val_hr_patches
        # lr_dataset[...] = val_lr_patches

        # TODO: save sample rates and stuff as attributes
