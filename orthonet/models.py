
import torch
from torch import nn
from torch.nn import functional as F

from orthonet import jacob

class AE(nn.Module):

    def __init__(self, input_size, latent_size):
        super(AE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.input_size, 1500)
        self.fc2 = nn.Linear(1500, 700)
        self.fc3 = nn.Linear(700, 300)
        self.fc4 = nn.Linear(300, self.latent_size)

        self.fc4b = nn.Linear(self.latent_size, 300)
        self.fc3b = nn.Linear(300, 700)
        self.fc2b = nn.Linear(700, 1500)
        self.fc1b = nn.Linear(1500, self.input_size)

        return

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc4(h3)

    def decode(self, z):
        h3b = F.relu(self.fc4b(z))
        h2b = F.relu(self.fc3b(h3b))
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

        self.fc1  = nn.Linear(self.input_size, 1500)
        self.fc2  = nn.Linear(1500, 700)
        self.fc31 = nn.Linear(700, 300)
        self.fc32 = nn.Linear(700, 300)
        self.fc41 = nn.Linear(300, self.latent_size)
        self.fc42 = nn.Linear(300, self.latent_size)

        self.fc4b = nn.Linear(self.latent_size, 300)
        self.fc3b = nn.Linear(300, 700)
        self.fc2b = nn.Linear(700, 1500)
        self.fc1b = nn.Linear(1500, self.input_size)

        return

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h31 = F.relu(self.fc31(h2))
        h32 = F.relu(self.fc32(h2))
        return self.fc41(h31), self.fc42(h32)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3b = F.relu(self.fc4b(z))
        h2b = F.relu(self.fc3b(h3b))
        h1b = F.relu(self.fc2b(h2b))
        return torch.sigmoid(self.fc1b(h1b))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.beta * KLD


