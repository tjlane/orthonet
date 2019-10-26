
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

    x : torch.Variable
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

    xr = x.squeeze() # duplicate to avoid any possible in-place issues
    n = x.size()[0]
    xr = x.repeat(n_outputs, 1)
    xr.requires_grad_(True)

    y = fxn(xr)
    I = torch.eye(n_outputs, device=xr.device)

    J = autograd.grad(y, xr,
                      grad_outputs=I, 
                      retain_graph=retain_graph, 
                      create_graph=True,  # for higher order derivatives
                      )

    return J[0]


def jacobian_grammian(fxn, x, n_outputs):
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
        The normalize Jacobian-Grammian G_ij^2 / (G_ii * G_jj)
    """

    J = jacobian(fxn, x, n_outputs)
    J[J == float('inf')] = 0

    n = x.size()[0]
    #assert J.shape == (n_outputs, n)

    G = torch.mm(torch.transpose(J, 0, 1), J) # Jacobian Grammian (outer product)
    h = torch.diag(G)             
    H = torch.ger(h,h)  # H_{ij} = G_{ii}

    JG = G.pow(2) / (H + 1e-8)

    return JG


def jg_loss(fxn, x, n_outputs, reduction='mean'):
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

    Returns
    ------- 
    loss : float
        The total loss over the dataset.
    """

    assert len(x.shape) == 2 # should be points x dims

    loss = 0.0
    for i in range(x.shape[0]):
        loss += torch.sum(jacobian_grammian(fxn, x[i], n_outputs))

    loss = loss / float(x.shape[1])

    if reduction == 'mean':
        loss = loss / float(x.shape[0])
    elif reduction == 'sum':
        pass
    else:
        raise ValueError('reduction must be {"sum", "mean"]')

    return loss


class JG_MSE_Loss:

    def __init__(self, beta=1.0, reduction='mean', track=True):
        """
        beta : float
            The beta parameter, higher means higher penalty for non-orthogonal
            output. Default = 1.0.

        reduction : str
            Either 'mean' or 'sum', just changes the normalization.

        track : bool
            Whether or not to internally store the piecewise losses
        """

        self.beta      = beta
        self.reduction = reduction
        self.track     = track

        if self.track:
            self._mse_losses = []
            self._ort_losses = []

        return


    @property
    def mse_losses(self):
        return np.array(self._mse_losses)


    @property
    def ort_losses(self):
        return np.array(self._ort_losses)


    def __call__(self, pred_y, y, x, net):
        """
        The orthogonal loss function.

            L_total = L_MSE + beta * L_ortho

        Parameters
        ----------
        pred_y : torch.Tensor
            The predicted output

        y : torch.Tensor
            Ground truth

        x : torch.Tensor
            The model input

        net : torch.nn.Model
            The neural network model

        Output
        -----
        loss : float
            The combination of the MSE and orthogonal loss.
        """

        mse = torch.sum((pred_y - y)**2)
        ort = jg_loss(net, x, y.shape[1], reduction=self.reduction)

        if self.track:
            self._mse_losses.append(mse.item())
            self._ort_losses.append(ort.item())

        return mse + self.beta * ort

