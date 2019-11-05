
import math

import torch
from torch import nn
from torch.nn import functional as F

from orthonet import jacob

def unif_init(tensor):
    stdv = 1. / math.sqrt(tensor.size(1))
    tensor.data.uniform_(-stdv, stdv)


class AE(nn.Module):
    def __init__(self, input_size, latent_size, tie_weights=True):
        super(AE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size
        self.tie_weights = tie_weights

        # TODO these should probably be exposed...
        self.hidden_dropout_p = 0.0 
        self.architecture = [self.input_size, 300, 300, 30, self.latent_size]
        self.activation   = F.elu

        self._init_params()

        return

    @property
    def n_layers(self):
        return len(self.architecture)

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_size))
        return self.decode(z)

    @staticmethod
    def loss_function(x, recon_x):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')
        return BCE

    def _init_params(self):

        self.encoder_params = []
        self.decoder_params = []

        for layer in range(self.n_layers-1):

            #print(layer, self.architecture[layer], '-->', self.architecture[layer+1])

            # encoder
            p = nn.Parameter(torch.zeros(self.architecture[layer+1],
                                         self.architecture[layer]))
            unif_init(p)
            setattr(self, 'param_e%d' % layer, p)
            self.encoder_params.append(p)

            # decoder
            if not self.tie_weights:
                p = nn.Parameter(torch.zeros(self.architecture[layer],
                                             self.architecture[layer+1]))
                unif_init(p)
                setattr(self, 'param_d%d' % layer, p)
                self.decoder_params.append(p)

        return

    def encode(self, x):
        for i,p in enumerate(self.encoder_params):
            if i == 0:
                l = self.activation( F.linear(x, p) )
            else:
                l = self.activation( F.linear(l, p) )
        return l

    def decode(self, z):

        if self.tie_weights:
            params = [p.t() for p in self.encoder_params[::-1]]
        else:
            params = self.decoder_params[::-1]

        for i,p in enumerate(params):
            if i == 0:
                l = self.activation( F.linear(z, p) )
            elif i == self.n_layers-2: # hidden counts as one extra...
                # output layer (if special)
                l = F.sigmoid( F.linear(l, p) )
            else:
                l = self.activation( F.linear(l, p) )
        return l


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

        self.hidden_dropout_p = 0.0

        self.shared = nn.Sequential(

          nn.Linear(self.input_size, 700),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),

          nn.Linear(700, 700),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU()
          )

        self.mu_branch = nn.Sequential(
          nn.Linear(700, 300),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),

          nn.Linear(300, self.latent_size),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU()
          )

        self.var_branch = nn.Sequential(
          nn.Linear(700, 300),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),
    
          nn.Linear(300, self.latent_size),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU()
          )


        self.decode = nn.Sequential(
          nn.Linear(self.latent_size, 300),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),

          nn.Linear(300, 700),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),

          nn.Linear(700, 700),
          nn.Dropout(self.hidden_dropout_p),
          nn.ELU(),

          nn.Linear(700, self.input_size),
          nn.Sigmoid()
          )

        return

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        return self.mu_branch(self.shared(x)), self.var_branch(self.shared(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.beta * KLD


