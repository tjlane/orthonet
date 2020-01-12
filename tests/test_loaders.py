
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


# H5Dataset

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

def test_preload():
    make_test_h5()
    ds = loaders.H5Dataset('tmp.h5', preload=True)
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
    assert len(ds) == 2
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


# DistributedDataLoader

def test_basic():

    rank = 0
    size = 1
    n_dpts = 47

    ds = np.random.randn(n_dpts, 2, 3)
    ddl = loaders.DistributedDataLoader(ds, rank, size, batch_size=1)

    assert ddl.data_range == (0, n_dpts)
    assert ddl.n_iter == n_dpts
    assert len(ddl) == n_dpts

    for i,b in enumerate(ddl):
        assert np.all(b == ds[i])

    return

def test_batch_size():

    rank = 0
    size = 1
    n_dpts = 47

    ds = np.random.randn(n_dpts, 2, 3)
    ddl = loaders.DistributedDataLoader(ds, rank, size, batch_size=3)

    assert len(ddl) == n_dpts // 3 + 1 # +1 for final batch

    for i,b in enumerate(ddl):
        assert np.all(b == ds[i*3:(i+1)*3])

    return

def test_distributive():

    size = 4
    n_dpts = 47
    ds = np.arange(n_dpts)

    ddl_1 = loaders.DistributedDataLoader(ds, 0, size, batch_size=1)
    ddl_2 = loaders.DistributedDataLoader(ds, 0, size, batch_size=2)

    for rank in range(size):

        ddl_1.rank = rank
        ddl_2.rank = rank

        st = 12 * rank
        sp = min(12 * (rank + 1), n_dpts)
        assert ddl_1.data_range == (st, sp)
        assert ddl_2.data_range == (st, sp)
        
        for i,b in enumerate(ddl_1):
            assert np.all(b == ds[st + i])

        for i,b in enumerate(ddl_2):
            assert np.all(b == ds[st + i*2 : st + (i+1)*2])

    return

def test_pin_memory():
    n_dpts = 47
    ds = np.arange(n_dpts)
    ddl = loaders.DistributedDataLoader(ds, 0, 1, batch_size=1, pin_memory=True)
    for i,b in enumerate(ddl):
        assert np.all(b == ds[i])
    return

def test_integration():

    make_test_h5()

    ds = loaders.H5Dataset('tmp.h5', preload=True)
    ddl = loaders.DistributedDataLoader(ds, 0, 2, batch_size=2)
    for i,b in enumerate(ddl):
        assert np.all(b[0] == i)
        assert np.all(b[1] == i+1)
        assert b.shape == (2,5,5)
    ds.close()

    ds2 = loaders.H5Dataset('tmp.h5', preload=False)
    ddl2 = loaders.DistributedDataLoader(ds2, 0, 2, batch_size=2)
    for i,b in enumerate(ddl2):
        assert np.all(b[0] == i)
        assert np.all(b[1] == i+1)
        assert b.shape == (2,5,5)
    ds2.close()
 
    rm_test_h5()
    return

def test_epoch_style_use():
    make_test_h5()
    ds2 = loaders.H5Dataset('tmp.h5', preload=False)
    ddl2 = loaders.DistributedDataLoader(ds2, 0, 2, batch_size=2)

    for epoch in range(5):
        for i,b in enumerate(ddl2):
            assert np.all(b[0] == i)
            assert np.all(b[1] == i+1)
            assert b.shape == (2,5,5)

    ds2.close()
    rm_test_h5()
    return

def test_PDDL():

    make_test_h5()

    ds = loaders.H5Dataset('tmp.h5', preload=True)
    ddl = loaders.PreloadingDDL(ds, 0, 2, batch_size=2)
    for i,b in enumerate(ddl):
        assert np.all(b[0] == i)
        assert np.all(b[1] == i+1)
        assert b.shape == (2,5,5)
    ds.close()

    rm_test_h5()
    return

