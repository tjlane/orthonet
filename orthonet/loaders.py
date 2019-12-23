
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms


class H5Dataset(Dataset):

    def __init__(self, in_file, shuffle=False, data_range=(0, None),
                 data_field='data', clip=(-1e8, 1e8)):
        """
        Parameters
        ----------
        in_file : str
            Path to the HDF5 file on disk.

        shuffle: bool
            Whether or not to shuffle the data randomly.

        data_range : (int, int)
            Limit the data access to a subset of the data, with min/max
            indices provided by this field.

        data_field : str
            The path to the data object inside the HDF5 file. Default: "/data".

        clip : (float, float)
            Values to clip the input data to. Default: (-1e8, 1e8).
        """

        super(H5Dataset, self).__init__()

        self.data_field = data_field
        self.f_handle = h5py.File(in_file, 'r', libver='latest', swmr=True)
        self.n_total = self.f_handle[self.data_field].shape[0]
        self.shape = self.f_handle[self.data_field].shape[1:]

        self.shuffle = shuffle
        self.set_data_range(data_range)

        # limit the min/max values in the images
        self.clip = clip

        return


    def __getitem__(self, index):

        if index < 0:
            raise NotImplementedError('negative indicies not yet supported')

        if self.shuffle:
            i = self._permutation[index]
        else:
            i = index + self.min_index

        assert ( i >= self.min_index )
        assert ( i <  self.max_index )
        item = self.f_handle[self.data_field][i]
        item = item.astype('float32').clip(*self.clip)

        return item


    def __len__(self):
        return self.n_datapoints


    def set_data_range(self, data_range):
        """
        Set a new limited data range. Useful for splitting a single
        HDF5 file into train/test sets.

        Parameters
        ----------
        data_range : (int, int)
            Limit the data access to a subset of the data, with min/max
            indices provided by this field.
        """

        # custom data range
        self.min_index = data_range[0]
        if data_range[1] is not None:
            self.n_datapoints = self.max_index - self.min_index
            self.max_index = data_range[1]
        else:
            self.n_datapoints = self.n_total
            self.max_index = self.n_total

        if self.shuffle:
            self._permutation = np.random.permutation(range(self.min_index,
                                                            self.max_index))
        return


    def close(self):
        self.f_handle.close()
        return



def load_data(data_file, batch_size, max_points=None,
              loader_kwargs={}, traintest_split=0.9,
              loader=DataLoader):
    """
    Load data into test/train loaders.
    """

    if data_file.split('/')[-1] == 'dsprites.h5':
        train_ds = H5Dataset(data_file, shuffle=True, data_field='imgs')
        test_ds  = H5Dataset(data_file, shuffle=True, data_field='imgs')
    else:
        train_ds = H5Dataset(data_file, clip=(0.0, 1.0)) # bot data
        test_ds  = H5Dataset(data_file, clip=(0.0, 1.0))

    if max_points is not None:
        size = min(max_points, len(train_ds))
    else:
        size = len(train_ds)
    split = int(size * traintest_split)

    print('\tTrain/Test: %d/%d' % (split, size-split))
    train_ds.set_data_range((0, split))
    test_ds.set_data_range((split, size))
    print('shps:', len(train_ds), train_ds.shape, '/', len(test_ds), test_ds.shape)

    train_loader = loader(train_ds,
                          batch_size=batch_size, 
                          shuffle=True, 
                          **loader_kwargs)

    test_loader = loader(test_ds,
                         batch_size=batch_size, 
                         shuffle=True, 
                         **loader_kwargs)

    data_shape = (len(train_ds) + len(test_ds),) + train_ds.shape

    return train_loader, test_loader, data_shape


def load_mnist(batch_size, loader_kwargs={}):
    """
    MNIST, serial, will download if not already
    """

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    # dataset is 60 000 training and 10 000 test 28x28 images
    data_shape = (60000 + 10000, 28, 28)

    return train_loader, test_loader, data_shape


