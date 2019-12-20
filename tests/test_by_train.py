
"""
Test by training a super simple network
"""

import torch
from torch import nn

from orthonet import jacob

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.f = nn.Sequential(
                     nn.Linear(4, 3),
                     nn.Sigmoid()
                    )
        return

    def forward(self, x):
        return self.f(x)


def test_training():

    n_out = 3

    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    x = torch.randn(2, 4)

    for i in range(3):
        y = m(x)
        loss = jacob.jg_loss(m, x, n_out)

        opt.zero_grad()
        loss.backward()
        opt.step()

if __name__ == '__main__':
    test_training()

