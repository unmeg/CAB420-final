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
        y = self.y_dataset[idx]

        x = torch.from_numpy(x)
        # y = torch.from_numpy(y)

        return (x, y)


if __name__ == '__main__':
    """
        Testing code and example usage
    """

    import time
    import sounddevice as sd
    import matplotlib.pyplot as plt

    dataset = HDF5PatchesDataset('train_pesq_large.hdf5')

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    scores = []
    for i, (x, y) in enumerate(dataloader):
        # print('\nBatch', i, 'Sizes:', x.size(), y.size())
        # print('min:', x.min(), 'max:', x.max(), 'mean:', x.mean())
        # print('min:', y.min(), 'max:', y.max(), 'mean:', y.mean())

        # print()
        # print(y.item())
        # sd.play(x[0, 0, :4095].numpy(), 16000)
        # sd.wait()

        # time.sleep(0.5)

        # sd.play(x[0, 0, 4096:8191].numpy(), 16000)
        # sd.wait()

        # time.sleep(2)

        score = y.item()
        scores.append(score)

        # plt.plot(x[0, 0, :].numpy())
        # plt.title("Score: {:.2f}".format(score / 10.0))
        # plt.show()

    scores = np.array(scores)
    plt.hist(scores, 50)
    plt.show()
