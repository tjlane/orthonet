
import os
import h5py
import numpy as np

from orthonet import loaders

def make_test_h5():
    rm_test_h5()
    f = h5py.File('tmp.h5', 'w')
    f['data'] = np.ones([4, 5, 5]) * np.arange(4)[:,None,None]
    f.close()
    return

def rm_test_h5():
    if os.path.exists('tmp.h5'):
        os.system('rm tmp.h5')
    return

# TODO use pytest to make the test file, load it, rm it only ONCE

def test_h5dataset_basics():
    make_test_h5()
    ds = loaders.H5Dataset('tmp.h5')
    item = ds[0]
    assert len(ds) == 4
    assert ds.shape == (5,5)
    assert np.all(item == np.zeros([5,5]))
    ds.close()
    rm_test_h5()
    return

def test_data_range():
    make_test_h5()
    ds = loaders.H5Dataset('tmp.h5')
    ds.set_data_range([2,4])
    item = ds[0]
    assert np.all(item == 2*np.ones([5,5]))
    item2 = ds[1]
    assert np.all(item2 == 3*np.ones([5,5]))
    ds.close()
    rm_test_h5()
    return

def test_shuffle():
    make_test_h5()
    ds = loaders.H5Dataset('tmp.h5', shuffle=True)
    item = ds[0]
    val = ds._permutation[0]
    assert np.all(item == val * np.ones([5,5]))
    ds.set_data_range([2,3]) # only one item, val=2
    item = ds[0]
    assert np.all(item == 2*np.ones([5,5]))
    ds.close()
    rm_test_h5()
    return

def test_clip():
    make_test_h5()
    ds = loaders.H5Dataset('tmp.h5', clip=(1,5))
    item = ds[0]
    assert np.all(item == np.ones([5,5]))
    ds.close()
    rm_test_h5()
    return




