
import torch
from torch import nn
from torch.nn import functional as F

from orthonet import jacob

class AE(nn.Module):

    def __init__(self, input_size, latent_size):
        super(AE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc2 = nn.Linear(400, 30)
        self.fc3 = nn.Linear(30, self.latent_size)

        self.fc3b = nn.Linear(self.latent_size, 30)
        self.fc2b = nn.Linear(30, 400)
        self.fc1b = nn.Linear(400, self.input_size)

        return

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, z):
        h2b = F.relu(self.fc3b(z))
        h1b = F.relu(self.fc2b(h2b))
        return torch.sigmoid(self.fc1b(h1b))

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_size))
        return self.decode(z)

    @staticmethod
    def loss_function(x, recon_x):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')
        return BCE


class OrthoAE(AE):

    def __init__(self, input_size, latent_size, beta=1.0):
        self._loss = jacob.JG_MSE_Loss(beta=beta)
        super(OrthoAE, self).__init__(input_size, latent_size)

    def loss_function(self, x, recon_x):
        return self._loss(recon_x, x.view(recon_x.shape), x.view(recon_x.shape), self)
        #return torch.sum((recon_x - x.view(recon_x.shape))**2) # TESTING


class VAE(nn.Module):

    def __init__(self, input_size, latent_size, beta=1.0):
        super(VAE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size

        self.beta = beta

        self.fc1  = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, self.latent_size)
        self.fc22 = nn.Linear(400, self.latent_size)
        self.fc3  = nn.Linear(self.latent_size, 400)
        self.fc4  = nn.Linear(400, self.input_size)

        return

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.beta * KLD


