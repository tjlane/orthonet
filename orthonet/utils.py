
import torch
import numpy as np

def binary_normalize(data, retscale=False, asrt=False):
    """
    Normalize data to be contained in [0,1), where the
    first dimension is considered an index of samples
    (ie the normalization is over all other axes)

        data' = (data + offset) / scale

    Parameters
    ----------
    data : np.ndarray
        (n, ...) shaped data, where n indexes the data samples

    retscale : bool
        True/False flag to return offset and scale parameters

    Returns
    -------
    data : np.ndarray
        The normalized data

    (offset, scale) : floats
        Optional parameters that can be used to re-scale data later
    """

    offset = np.min(data, axis=(1,2))
    data = data - offset[:,None,None]
    scale = np.max(data, axis=(1,2))
    data = data / (scale[:,None,None] + 1e-8)

    if asrt:
        assert np.all(np.min(data, axis=(1,2)) == 0.0), np.min(data)
        assert np.all(np.max(data, axis=(1,2)) == 1.0), np.min(np.max(data, axis=(1,2)))

    if retscale:
        return data, (offset, scale)
    else:
        return data


