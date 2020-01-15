
import h5py
import numpy as np
from math import ceil

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils import pin_memory as pm

from torchvision import datasets, transforms


class H5Dataset(Dataset):

    def __init__(self, in_file, data_range=(0, None),
                 data_field='data', clip=None, dtype='float32'):
        """
        Parameters
        ----------
        in_file : str
            Path to the HDF5 file on disk.

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

        self.set_data_range(data_range) # sets _data
        self.clip = clip

        return


    def __getitem__(self, index):
        # NOTE index may be a slice object

        if isinstance(self._data, np.ndarray):
            offset = 0
        else:
            offset = self.min_index

        if type(index) is int:
            i = index + offset # ints, arrays
            if index < 0:
                raise NotImplementedError('negative indicies not supported')
        elif type(index) is slice:
                i = slice(index.start + offset,
                          index.stop  + offset)
        else:
            raise TypeError('int and slice indices are only safe types as of now')
            
        item = self._data[i]
        if item.dtype != self.dtype:
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

        self._data = self.f_handle[self.data_field]

        return


    def preload(self):
        self._data = self.f_handle[self.data_field][self.min_index:self.max_index].astype(self.dtype)
        return


    def close(self):
        self.f_handle.close()
        return


class DistributedDataLoader:

    def __init__(self, dataset, rank, size, batch_size=1, pin_memory=False):

        self.dataset = dataset
        self.rank = rank
        self.size = size
        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.epoch = 0

        return


    @property
    def data_range(self):

        if isinstance(self.dataset, H5Dataset):
            # len here changes if we set range
            n_total = self.dataset.n_total
        else:
            n_total = len(self.dataset) # np.array, etc

        # the data to pull from the dataset
        # note: this is not the batch range
        slc = (self.rank + self.epoch) % self.size
        slc_size = ceil(n_total / self.size)
        start = slc * slc_size
        stop  = min((slc+1) * slc_size, n_total)
        return start, stop


    @property
    def n_iter(self):
        start, stop = self.data_range
        return ceil( (stop - start) / self.batch_size )


    def __iter__(self):

        start, stop = self.data_range

        for i in range(self.n_iter):
            if self.batch_size == 1:
                data = self.dataset[start + i]
            else:
                b_start = start + i * self.batch_size
                b_stop  = min(start + (i+1) * self.batch_size, stop)
                s = slice(b_start, b_stop)
                data = self.dataset[s]

            if not type(data) == torch.Tensor:
                data = torch.tensor(data)

            if self.pin_memory:
                data = pm.pin_memory(data)

            yield data

    def __len__(self):
        return self.n_iter


    def set_epoch(self, epoch):
        self.epoch = epoch
        return


class PreloadingDDL(DistributedDataLoader):

    def __init__(self, *args, **kwargs):
        super(PreloadingDDL, self).__init__(*args, **kwargs)
        self._data_range = None
        self._preload()

        if not isinstance(self.dataset, H5Dataset):
            raise TypeError('PreloadingDDL requires an H5Dataset to function')

        return


    def _preload(self):

        start, stop = self.data_range

        # if the data range is new, preload data
        if self._data_range != (start, stop):
            self.dataset.set_data_range((start, stop))
            self.dataset.preload()
            self._data_range = (start, stop)

        return


    def __iter__(self):
        """
        This works with H5Dataset's preload capability to load a single
        rank/epoch datarange into memory at once.

        * The main difference is we set the H5Dataset range, which
          means that __getitem__ indexing will always start at 0
        """

        start, stop = self.data_range
        self._preload()

        for i in range(self.n_iter):
            if self.batch_size == 1:
                data = self.dataset[i]
            else:
                b_start = i * self.batch_size
                b_stop  = min((i+1) * self.batch_size, len(self.dataset))
                s = slice(b_start, b_stop)
                data = self.dataset[s]

            if not type(data) == torch.Tensor:
                data = torch.tensor(data)

            if self.pin_memory:
                data = pm.pin_memory(data)

            yield data



# ^^^ general useful code above ^^^
# --- below here is project/cluster specific ----------------------------------


def load_bot(data_file, batch_size, max_points=None,
             loader_kwargs={}, traintest_split=0.9):
    """
    Load bot simulation data into test/train loaders.
    """

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

    
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size, 
                              **loader_kwargs)

    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
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
        datasets.MNIST('./data', train=False, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    # dataset is 60 000 training and 10 000 test 28x28 images
    data_shape = (60000 + 10000, 28, 28)

    return train_loader, test_loader, data_shape


def load_dsprites(batch_size, rank=0, size=1, preload=False, pin_memory=True):

    data_file = '/scratch/tjlane/dsprites.h5'
    train_ds = H5Dataset(data_file, data_field='imgs_shuffled_train')
    test_ds  = H5Dataset(data_file, data_field='imgs_shuffled_test')

    if preload:
        DDL = PreloadingDDL
    else:
        DDL = DistributedDataLoader
    
    # each rank gets a unique training set
    train_loader = DDL(train_ds, rank, size, batch_size=batch_size, pin_memory=pin_memory)

    # but identical test sets
    test_loader = DDL(test_ds, 0, 1, batch_size=batch_size, pin_memory=pin_memory)

    print('DataLoader rank %d :: %s :: preload=%d || train : %d / test : %d'
          '' % (rank, str(train_loader.data_range), int(preload), len(train_ds), len(test_ds)))

    data_shape = (len(train_ds) + len(test_ds),) + train_ds.shape

    return train_loader, test_loader, data_shape 

