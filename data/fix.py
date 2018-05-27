import h5py
import numpy as np

num_classes = 50
pesq_sr = 16000
input_hdf5 = 'train_vctk_patches.hdf5'
output_hdf5 = 'train_pesq.hdf5'


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

h5_file = h5py.File('train_pesq.hdf5', 'r')
hr_dataset = h5_file['y']

a = np.array(hr_dataset).copy()

h5_file.close()

print(a)
print(a.shape)

with h5py.File('hdf5new_train_pesq.', 'w') as f:
    y = f.create_dataset("y2", a.shape, dtype=np.int32)
    y[...] = a

