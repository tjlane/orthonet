
import h5py
import numpy as np
from math import ceil

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms


class H5Dataset(Dataset):

    def __init__(self, in_file, shuffle=False, data_range=(0, None),
                 data_field='data', clip=None, preload=False, dtype='float32'):
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
            Values to clip the input data to. Default: None.
        """

        super(H5Dataset, self).__init__()

        self.dtype = dtype
        self.data_field = data_field
        self.f_handle = h5py.File(in_file, 'r', libver='latest', swmr=True)

        self.preload = preload
        self.shuffle = shuffle
        self.set_data_range(data_range)
        self.clip = clip

        if self.preload:
            self._data = self.f_handle[self.data_field][self.min_index:self.max_index].astype(self.dtype)
        else:
            self._data = self.f_handle[self.data_field]

        return


    def __getitem__(self, index):

        # NOTE index may be a slice object

        if type(index) == int:
            if index < 0:
                raise NotImplementedError('negative indicies not yet supported')

        if self.shuffle:
            i = self._permutation[index]
            # with current h5py interface, will be an error IF all:
            #  * self.shuffle = True
            #  * batch_size > 1
            #  * self.preload = False
            #     because h5py does not allow integer indexing
            #     do not fix for now, as update to h5py is coming

        else:
            if type(index) == slice:
                i = slice(index.start - self.min_index,
                          index.stop  - self.min_index)
            else:
                i = index + self.min_index # ints, arrays
            
        item = self._data[i]
        item = item.astype(self.dtype)
        if self.clip:
            item = item.clip(*self.clip)

        return item


    def __len__(self):
        return min(self.max_index - self.min_index, self.n_total)


    @property
    def n_total(self):
        return self.f_handle[self.data_field].shape[0]


    @property
    def shape(self):
        return self.f_handle[self.data_field].shape[1:]


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
            self.max_index = data_range[1]
        else:
            self.max_index = self.n_total

        if self.shuffle:
            self._permutation = np.random.permutation(range(self.min_index,
                                                            self.max_index))

        if self.preload:
            self._data = self.f_handle[self.data_field][self.min_index:self.max_index].astype(self.dtype)

        return


    def close(self):
        self.f_handle.close()
        return


class DistributedDataLoader:

    def __init__(self, dataset, rank, size, batch_size=1):

        self._dataset = dataset
        self.rank = rank
        self.size = size
        self.batch_size = batch_size

        self.epoch = 0

        return


    @property
    def data_range(self):
        # the data to pull from the dataset
        # note: this is not the batch range
        slc = (self.rank + self.epoch) % self.size
        slc_size = ceil(len(self._dataset) / self.size)
        start = slc * slc_size
        stop  = min((slc+1) * slc_size, len(self._dataset))
        return start, stop


    @property
    def n_iter(self):
        start, stop = self.data_range
        return ceil( (stop - start) / self.batch_size )


    def __iter__(self):

        start, stop = self.data_range

        for i in range(self.n_iter):
            if self.batch_size == 1:
                yield self._dataset[start + i]
            else:
                b_start = start + i * self.batch_size
                b_stop  = min(start + (i+1) * self.batch_size, stop)
                s = slice(b_start, b_stop)
                yield self._dataset[s]


    def __len__(self):
        return self.n_iter


    def set_epoch(self, epoch):
        self.epoch = epoch
        return


class PreloadingDDL(DistributedDataLoader):

    def __init__(self, *args, **kwargs):
        super(PreloadingDDL, self).__init__(*args, **kwargs)

        if not isinstance(self._dataset, H5Dataset):
            raise TypeError('PreloadingDDL requires an H5Dataset to function')

        if not self._dataset.preload == True:
            raise ValueError('PreloadingDDL H5Dataset.preload = True')

        return


    def __iter__(self):
        """
        This works with H5Dataset's preload capability to load a single
        rank/epoch datarange into memory at once.

        * The main difference is we set the H5Dataset range, which
          means that __getitem__ indexing will always start at 0
        """

        start, stop = self.data_range
        self._dataset.set_data_range((start, stop))

        for i in range(self.n_iter):
            if self.batch_size == 1:
                yield self._dataset[i]
            else:
                b_start = i * self.batch_size
                b_stop  = min((i+1) * self.batch_size, len(self._dataset))
                s = slice(b_start, b_stop)
                yield self._dataset[s]



def load_data(data_file, batch_size, max_points=None,
              loader_kwargs={}, traintest_split=0.9,
              dist_sampler_kwargs=None):
    """
    Load data into test/train loaders.
    """

    if data_file.split('/')[-1] == 'dsprites.h5':
        train_ds = H5Dataset(data_file, data_field='imgs_shuffled', preload=False)
        test_ds  = H5Dataset(data_file, data_field='imgs_shuffled', preload=False)
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

    
    if dist_sampler_kwargs:
       train_sampler = DistributedSampler(train_ds,
                                          **dist_sampler_kwargs)
       test_sampler  = DistributedSampler(test_ds,
                                          **dist_sampler_kwargs)
    else:
        train_sampler = None
        test_sampler  = None


    train_loader = DataLoader(train_ds,
                              batch_size=batch_size, 
                              sampler=train_sampler,
                              **loader_kwargs)

    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             **loader_kwargs)

    data_shape = (len(train_ds) + len(test_ds),) + train_ds.shape

    return train_loader, test_loader, data_shape


def load_mnist(batch_size, loader_kwargs={}, dist_sampler_kwargs=None):
    """
    MNIST, serial, will download if not already
    """

    if dist_sampler_kwargs is None:
        raise NotImplementedError('mnist parallel not in yet (but is easy)')

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


