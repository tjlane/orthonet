
import torch
from torch import autograd
import numpy as np

@torch.enable_grad()
def jacobian(fxn, x, n_outputs, retain_graph=True):
    """
    Compute the Jacobian of a function.

    Parameters
    ----------
    fxn : function
        The pytorch function (e.g. neural network object)

    x : torch.Tensor
        The input point at which to evaluate J

    n_outputs : int
        The expected dimension of the output of net (dimension of range)

    Returns
    -------
    J : torch.Tensor
        The Jacobian d(fxn) / dx | x, shape (n_outputs, n)
    """

    # the basic idea is to create N copies of the input
    # and then ask for each of the N dimensions of the
    # output... this allows us to compute J with pytorch's
    # jacobian-vector engine

    # expand the input, one copy per output dimension
    n_outputs = int(n_outputs)
    repear_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repear_arg)
    xr.requires_grad_(True)

    # both y and I are shape (n_outputs, n_outputs)
    #  checking y shape lets us report something meaningful
    y = fxn(xr).view(n_outputs, -1)

    if y.size(1) != n_outputs: 
        raise ValueError('Function `fxn` does not give output '
                         'compatible with `n_outputs`=%d, size '
                         'of fxn(x) : %s' 
                         '' % (n_outputs, y.size(1)))
    I = torch.eye(n_outputs, device=xr.device)

    J = autograd.grad(y, xr,
                      grad_outputs=I,
                      retain_graph=retain_graph,
                      create_graph=True,  # for higher order derivatives
                      )

    return J[0]


def jacobian_grammian(fxn, x, n_outputs, normalize=False):
    """
    Compute the Grammian matrix of the Jacobian of `fxn` at `x`.

    Parameters
    ----------
    fxn : nn.Model
        The pytorch function (e.g. neural network object)

    x : torch.Variable
        The input point at which to evaluate J

    n_outputs : int
        The expected dimension of the output of net (dimension of range)

    Returns
    -------
    GJ : torch.Tensor
        The Jacobian-Grammian J^T * J ( size n x n, n=size(flat(x))) )
    """

    J = jacobian(fxn, x, n_outputs)
    Jc = J.clamp(-1*2**31, 2**31) # prevent numbers that are too large

    #n = x.size(0)
    #assert J.shape == (n_outputs, n)

    G = torch.mm(torch.transpose(Jc, 0, 1), Jc) # Jacobian Grammian (outer product)

    if normalize:
        h = torch.diag(G)             
        H = torch.ger(h,h)  # H_{ij} = G_{ii}
        JG = G.pow(2) / (H + 1e-8)
        return JG
    else:
        return G


def jg_loss(fxn, x, n_outputs, reduction='mean', diagonal_weight=1.0):
    """
    Compute the aggregated Jacobian-Grammian loss over a dataset `x` under
    model `fxn`.

    Parameters
    ----------
    fxn : nn.Model
        The pytorch function (e.g. neural network object)

    x : torch.Variable
        The input point at which to evaluate J

    n_outputs : int
        The expected dimension of the output of net (dimension of range)

    reduction : str
        Either 'mean' or 'sum', just changes the normalization.

    diagonal_weight : float
        How much to weight the diagonal (self) values of the
        Jacobian Grammian

    Returns
    ------- 
    loss : float
        The total loss over the dataset.
    """

    assert len(x.shape) == 2 # should be points x dims

    jg_accum = torch.zeros(x.size(1), x.size(1), device=x.device) 
    for i in range(x.size(0)):
        jg_accum += jacobian_grammian(fxn, x[i], n_outputs)

    jg2 = jg_accum.pow(2) 
    loss = jg2.sum() + (diagonal_weight - 1.0) * jg2.diag().sum()
    loss = torch.sqrt(loss) / float(x.size(1) ** 2)

    if reduction == 'mean':
        loss = loss / float(x.size(0))
    elif reduction == 'sum':
        pass
    else:
        raise ValueError('reduction must be {"sum", "mean"}')

    return loss


def jf_loss(fxn, x, n_outputs, reduction='mean'):
    """
    Compute the Frobenius norm loss, as per a contractive AE.

    Parameters
    ----------
    fxn : nn.Model
        The pytorch function (e.g. neural network object)

    x : torch.Variable
        The input point at which to evaluate J

    n_outputs : int
        The expected dimension of the output of net (dimension of range)

    reduction : str
        Either 'mean' or 'sum', just changes the normalization.

    Returns
    ------- 
    loss : float
        The total loss over the dataset.

    Citations
    ---------
    .[1] Rifai, Vincent, Muller, Glorot, Bengio ICML (2011).
    """

    assert len(x.shape) == 2 # should be points x dims

    n = x.size(1)
    jf_accum = torch.zeros(n_outputs, n, device=x.device)
    for i in range(x.size(0)):
        jf_accum += jacobian(fxn, x[i], n_outputs)

    loss = torch.sqrt( jf_accum.pow(2).sum() ) / float(n_outputs * n)

    if reduction == 'mean':
        loss = loss / float(x.size(0))
    elif reduction == 'sum':
        pass
    else:
        raise ValueError('reduction must be {"sum", "mean"}')

    return loss


