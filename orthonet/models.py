

import math

import torch
from torch import nn
from torch.nn import functional as F

from orthonet import jacob


def unif_init(tensor):
    stdv = 1. / math.sqrt(tensor.size(1))
    tensor.data.uniform_(-stdv, stdv)


class OpenDataParallel(nn.DataParallel):
    """
    DataParallel hides the module's methods to avoid name conflicts
    This class simply exposes a few key methods from the base classes
    """

    def encode(self, arg):
        return self.module.encode(arg)

    def decode(self, arg):
        return self.module.decode(arg)

    def loss_function(self, *args):
        return self.module.loss_function(*args)



class ResFC(nn.Module):
    """
    A fully connected two-stage residual layer, with the following goodies:
        * batchnorm
        * dropout
        * uses a linear (affine) transform to line up mismatched input/output
          dimensions
    """
    def __init__(self, in_size, out_size, activation=F.relu, dropout_p=0.5):
        super(ResFC, self).__init__()

        self.in_size  = in_size
        self.out_size = out_size
        self.activation = activation
        self.dropout_p  = dropout_p

        self.t1 = nn.Linear(self.in_size, self.out_size, bias=False)
        self.t2 = nn.Linear(self.out_size, self.out_size, bias=False)
        self.bn = nn.BatchNorm1d(self.out_size)
        self.dp = nn.Dropout(p=dropout_p, inplace=False)

        if self.in_size != self.out_size:
            self.affine = nn.Linear(self.in_size, self.out_size, bias=False)
        else:
            self.affine = nn.Identity()

        return

    def forward(self, x):
        r  = self.affine(x)
        Fx = self.t2(self.activation(self.t1(x)))
        nm = self.bn(Fx + r)
        z  = self.dp(nm)
        return z


class SigmoidOutput(nn.Module):
    def __init__(self, in_size, out_size):
        super(SigmoidOutput, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class AE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size

        self.encode_conv = nn.Sequential(
                            # input size is 1 x 33 x 33
                            nn.Conv2d(1, 4, 4, stride=2, padding=4, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 4 x 16 x 16
                            nn.Conv2d(4, 16, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 16 x 8 x 8
                            nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 32 x 4 x 4
                            nn.Conv2d(32, 64, 4, stride=2, padding=0, bias=False),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(0.2, inplace=True)
                            # --> into FC is 64 x 1 x 1
                          )
        self.encode_fc  = nn.Sequential(
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64),

                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, self.latent_size),
                            nn.LeakyReLU(0.2, inplace=True),
                          )

        self.decode_fc   = nn.Sequential(
                            nn.Linear(self.latent_size, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64),

                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64)
                          )
        self.decode_conv = nn.Sequential(

                            # input is 64 x 1 x 1
                            nn.ConvTranspose2d(64, 32, 4, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.2, inplace=True),

                            # size 32 x 4 x 4
                            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU(0.2, inplace=True),

                            # size 16 x 8 x 8
                            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(0.2, inplace=True),

                            # size 8 x 4 x 4
                            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(4, 1, 4, stride=1, padding=1, bias=False),
                            nn.Sigmoid()
                        )

        return

    def encode(self, x):
        conv_out = self.encode_conv(x.view(-1,1,33,33))
        conv_out_flat = conv_out.squeeze()
        return self.encode_fc(conv_out_flat)

    def decode(self, z):
        fc_out = self.decode_fc(z)
        fc_out_expand = fc_out.unsqueeze(-1).unsqueeze(-1)
        conv_out = self.decode_conv(fc_out_expand)
        return conv_out.squeeze()

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_size))
        return self.decode(z)

    @staticmethod
    def loss_function(x, recon_x):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')
        return BCE



class OrthoAE(AE):

    def __init__(self, input_size, latent_size, beta=1.0):
        super(OrthoAE, self).__init__(input_size, latent_size)
        self._loss = jacob.JG_MSE_Loss(beta=beta)

    def loss_function(self, x, recon_x):
        return self._loss(recon_x, x.view(recon_x.shape), x.view(recon_x.shape), self)


class VAE(nn.Module):

    def __init__(self, input_size, latent_size, beta=1.0):
        super(VAE, self).__init__()

        self.input_size  = input_size
        self.latent_size = latent_size

        self.beta = beta

        # encoder
        self.shared     = nn.Sequential(
                            # input size is 1 x 33 x 33
                            nn.Conv2d(1, 4, 4, stride=2, padding=4, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 4 x 16 x 16
                            nn.Conv2d(4, 16, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 16 x 8 x 8
                            nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.2, inplace=True),

                            # current size is 32 x 4 x 4
                            nn.Conv2d(32, 64, 4, stride=2, padding=0, bias=False),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(0.2, inplace=True)
                            # --> into FC is 64 x 1 x 1
                          )
        self.mu_branch  = nn.Sequential(
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64),

                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, self.latent_size),
                            nn.LeakyReLU(0.2, inplace=True),
                          )
        self.var_branch = nn.Sequential(
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64),

                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, self.latent_size),
                            nn.LeakyReLU(0.2, inplace=True),
                          )


        # decoder
        self.decode_fc   = nn.Sequential(
                            nn.Linear(self.latent_size, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64),

                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.BatchNorm1d(64)
                          )
        self.decode_conv = nn.Sequential(

                            # input is 64 x 1 x 1
                            nn.ConvTranspose2d(64, 32, 4, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.2, inplace=True),
                    
                            # size 32 x 4 x 4
                            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU(0.2, inplace=True),
                    
                            # size 16 x 8 x 8
                            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(0.2, inplace=True),
                    
                            # size 8 x 4 x 4
                            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, bias=False),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(4, 1, 4, stride=1, padding=1, bias=False),
                            nn.Sigmoid()
                        )

        return

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        conv_out = self.shared(x.view(-1,1,33,33))
        conv_out_flat = conv_out.squeeze()
        return self.mu_branch(conv_out_flat), self.var_branch(conv_out_flat)

    def decode(self, z):
        fc_out = self.decode_fc(z)
        fc_out_expand = fc_out.unsqueeze(-1).unsqueeze(-1)
        conv_out = self.decode_conv(fc_out_expand)
        return conv_out.squeeze()

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.beta * KLD


