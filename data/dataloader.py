import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler

import h5py
import numpy as np


class HDF5PatchesDataset(Dataset):
    def __init__(self, hdf5_path):

        self.h5_file = h5py.File(hdf5_path, 'r')
        self.x_dataset = self.h5_file['x']
        self.y_dataset = self.h5_file['y']

    def __len__(self):
        return self.x_dataset.shape[0]

    def __getitem__(self, idx):

        x = self.x_dataset[idx, :, :]
        y = self.y_dataset[idx, :]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return (x, y)


if __name__ == '__main__':
    """
        Testing code and example usage
    """

    import time
    import sounddevice as sd
    import matplotlib.pyplot as plt

    dataset = HDF5PatchesDataset('train_pesq.hdf5')

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    for i, (x, y) in enumerate(dataloader):
        print('\nBatch', i, 'Sizes:', x.size(), y.size())
        print('min:', x.min(), 'max:', x.max(), 'mean:', x.mean())
        print('min:', y.min(), 'max:', y.max(), 'mean:', y.mean())

        sd.play(x[0, 0, :].numpy(), 16000)
        sd.wait()

        time.sleep(2)
