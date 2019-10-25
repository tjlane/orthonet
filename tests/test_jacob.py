
import torch
import numpy as np

from orthonet import jacob


# -----------------------------------------------

# some analytical examples for testing

def y(x):
    
    # y(x) = [x0, x1^2, x0 * x1]
    
    if len(x.shape) == 1:
        x=x.repeat(1,1)
        
    y = torch.zeros(x.shape[0], 3)
    
    y[:,0] = x[:,0]
    y[:,1] = x[:,1]**2
    y[:,2] = x[:,0] * x[:,1]
    
    return y


def Jy(x):
    # only works for a single point x
    J = torch.tensor([[1.0,  0.0],
                      [0.0,  2.0*x[1]],
                      [x[1], x[0]]])
    return J

# -----------------------------------------------


def test_jacobian():
    for i in range(10):
        x = torch.randn(2)
        assert torch.allclose( Jy(x), jacob.jacobian(y, x, 3))


def test_jacobian_grammian():
    pass


def test_jg_loss():
    pass


def test_jg_loss_function():
    pass


    
if __name__ == '__main__':
    test_jacobian()


