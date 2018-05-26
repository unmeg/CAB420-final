import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler

import os
import h5py
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate


def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    x_sp = x_sp.reshape(1, x_sp.shape[0])

    return x_sp


def wav_collate(batch):
    """
        Used by dataloader internally to arrange batches nicely
        Returns a tuple (x, y) where x is a Tensor of batch_size downsampled audio waveforms (LR)
        and y is a Tensor of batch_size original audio waveforms (HR)
    """
    batch_size = len(batch)
    x_lr = torch.zeros(batch_size, batch[0][0].shape[0])
    x_hr = torch.zeros(batch_size, batch[0][1].shape[0])

    for b in range(batch_size):
        x_lr[b, :] = torch.from_numpy(batch[b][0].copy())
        x_hr[b, :] = torch.from_numpy(batch[b][1].copy())

    x_lr.unsqueeze_(1)
    x_hr.unsqueeze_(1)

    return x_lr, x_hr


def hdf5_collate(batch):
    """
        Used by dataloader internally to arrange batches nicely
        Returns a tuple (x, y) where x is a Tensor of batch_size downsampled audio waveforms (LR)
        and y is a Tensor of batch_size original audio waveforms (HR)
    """
    batch_size = len(batch)

    x_lr = torch.zeros(batch_size, 1, batch[0][0].shape[1])
    x_hr = torch.zeros(batch_size, 1, batch[0][1].shape[1])

    for b in range(batch_size):
        x_lr[b, :, :] = batch[b][0]
        x_hr[b, :, :] = batch[b][1]

    return x_lr, x_hr


class RangeSampler(sampler.Sampler):
    """
        This is a thing to specify a length and offset for dataset.
        Note: cannot use shuffle=True with a sampler

        Usage is like:

        # create dataset
        dataset = WAVFilesDataset(dir, Downsample(2))

        # load in 100 training examples, starting at index 0
        training_dataloader = DataLoader(dataset, batch_size=2, num_workers=2,
            collate_fn=wav_collate, sampler=RangeSampler(100, 0))

        # load in 10 test examples, starting at index 100 (from end of training examples)
        test_dataloader = DataLoader(dataset, batch_size=2, num_workers=2,
            collate_fn=wav_collate, sampler=RangeSampler(10, 100))
    """

    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InterpolateUpsample(object):
    def __init__(self, upscaling_factor):
        self.upscaling_factor = upscaling_factor

    def __call__(self, sample):
        x, y = sample
        x = upsample(x, self.upscaling_factor)
        return (x, y)


class Downsample(object):
    """
        This class is a 'Transform' used by Dataset objects to do any pre-processing
        on data before it gets batched-up by the dataloader.

        args: downsampling_factor - factor to downsample by
    """

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def __call__(self, sample):
        x_lr, x_hr = sample

        x_lr = signal.decimate(x_lr, self.downsampling_factor)

        return (x_lr, x_hr)


class WAVFilesDataset(Dataset):
    """
        This Dataset takes a directory as an argument and will load all
        .wav files in that directory. Without a transform specified, it will return (x, y) as both HR.

        This can easily be extended to load LR and HR from a directory depending on filename or sub-folder or something...
        TODO: wait and see what structure of VCTK dataset looks like before adding this functionality
    """

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform

        self.filenames = []

        for filename in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, filename)) and filename.lower().endswith('.wav'):
                self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        _, data = wavfile.read(os.path.join(self.data_dir, self.filenames[idx]))

        sample = (data, data)

        if self.transform:
            sample = self.transform(sample)

        return sample


class HDF5PatchesDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):

        self.h5_file = h5py.File(hdf5_path, 'r')
        self.hr_dataset = self.h5_file['hr']
        self.lr_dataset = self.h5_file['lr']
        self.transform = transform

    def __len__(self):
        return self.hr_dataset.shape[0]

    def __getitem__(self, idx):

        lr = self.lr_dataset[idx, :, :]
        hr = self.hr_dataset[idx, :, :]

        if self.transform is not None:
            lr, hr = self.transform((lr, hr))

        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return (lr, hr)


if __name__ == '__main__':
    """
        Testing code and example usage
    """

    import time
    import sounddevice as sd
    import matplotlib.pyplot as plt

    dataset = HDF5PatchesDataset('../data/val_patches.hdf5')

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

    for i, (x, y) in enumerate(dataloader):
        print('\nBatch', i, 'Sizes:', x.size(), y.size())
        print('min:', x.min(), 'max:', x.max(), 'mean:', x.mean())
        print('min:', y.min(), 'max:', y.max(), 'mean:', y.mean())

        sd.play(x[0, 0, :].numpy(), 24000)
        sd.wait()

        time.sleep(0.5)

        sd.play(y[0, 0, :].numpy(), 24000)
        sd.wait()

        time.sleep(2)
