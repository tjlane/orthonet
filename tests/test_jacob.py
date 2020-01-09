
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


def test_rev_jacobian():
    for i in range(10):
        x = torch.randn(2)
        J = jacob.jacobian(y, x, 3, ad_mode='rev')
        assert torch.allclose( Jy(x), J )


def test_fwd_jacobian():
    for i in range(10):
        x = torch.randn(2)
        J = jacob.jacobian(y, x, 3, ad_mode='fwd')
        print(J)
        print(Jy(x))
        assert torch.allclose( Jy(x), J )


def test_jacobian_grammian():
    for i in range(10):
        x = torch.randn(2)
        J = Jy(x)
        assert torch.allclose( torch.mm(J.t(), J), jacob.jacobian_grammian(y, x, 3))
   

if __name__ == '__main__':
    test_fwd_jacobian()
    #test_jacobian_grammian()

