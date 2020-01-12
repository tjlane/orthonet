
import torch
from torch import autograd
import numpy as np


def jacobian(fxn, x, n_outputs, retain_graph=True, ad_mode='fwd'):
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

    ad_mode : str
        One of {'fwd', 'rev'} indicating the mode for automatic differentiation.
        Results are identical for both, but for the Jacobian of a function
        mapping R^n --> R^m, forward mode is generally faster if n < m, reverse
        if n > m.

    Returns
    -------
    J : torch.Tensor
        The Jacobian d(fxn) / dx | x, shape (n_outputs, n)
    """

    if ad_mode == 'fwd':
        J = _fwd_jacobian(fxn, x, n_outputs, retain_graph)
    elif ad_mode == 'rev':
        J = _rev_jacobian(fxn, x, n_outputs, retain_graph)
    else:
        raise ValueError('Invalid autodiff mode: %s. Choose "fwd" or "rev"'
                         '' % ad_mode)

    return J


@torch.enable_grad()
def _rev_jacobian(fxn, x, n_outputs, retain_graph):
    """
    the basic idea is to create N copies of the input
    and then ask for each of the N dimensions of the
    output... this allows us to compute J with pytorch's
    jacobian-vector engine
    """

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


@torch.enable_grad()
def _fwd_jacobian(fxn, x, n_outputs, retain_graph):
    """
    This implementation is very similar to the above, but with
    one twist. To implement a forward-mode AD with rev-mode
    calls, we first compute the rev-mode VJP for one vector (v)
    then we call d/dv(VJP) `n_outputs` times, one per basis vector,
    to obtain the Jacobian.
    
    This should be faster if `n_outputs` > "n_inputs"

    References
    ----------
    .[1] https://j-towns.github.io/2017/06/12/A-new-trick.html
         (Thanks to Jamie Townsend for this awesome trick!)
    """

    xd = x.detach().requires_grad_(True)
    n_inputs = int(xd.size(0))

    # first, compute *any* VJP 
    v = torch.ones(n_outputs, device=x.device, requires_grad=True)
    y = fxn(xd.view(1,n_inputs)).view(n_outputs)

    if y.size(0) != n_outputs:
        raise ValueError('Function `fxn` does not give output '
                         'compatible with `n_outputs`=%d, size '
                         'of fxn(x) : %s'
                         '' % (n_outputs, y.size(0)))

    vjp = torch.autograd.grad(y, xd, grad_outputs=v, 
                              create_graph=True,
                              retain_graph=retain_graph)[0]
    assert vjp.shape == (n_inputs,)

    # TODO somehow the repeat trick does not work anymore
    #      now that we have to take derivatives wrt v
    #      so loop over basis vectors and compose jacobian col by col

    I = torch.eye(n_inputs, device=x.device)
    J = []
    for i in range(n_inputs):
        Ji = autograd.grad(vjp, v,
                          grad_outputs=I[i],
                          retain_graph=retain_graph,
                          create_graph=True,  # for higher order derivatives
                          )
        J.append(Ji[0])

    return torch.stack(J).t()



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


