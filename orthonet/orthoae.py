
import torch
from torch import nn
from torch.nn import functional as F

FIELD_DIMENSION = 1089


class OrthoEnc(nn.Module):
    def __init__(self):
        super(OrthoEnc, self).__init__()

        self.fc1 = nn.Linear(FIELD_DIMENSION, 400)
        self.fc2 = nn.Linear(400, 4)
        self.fc3 = nn.Linear(4, 400)
        self.fc4 = nn.Linear(400, FIELD_DIMENSION)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, FIELD_DIMENSION))
        return self.decode(z)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, FIELD_DIMENSION), reduction='sum')
    return BCE


