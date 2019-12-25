
from os.path import join as pjoin

import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision.utils import save_image


def plot_loss_curves(train_loss, test_loss, save=False):
    """
    save : str
        path to save the image to, or False if no save desired
    """
    
    plt.figure()

    if len(train_loss.shape) == 1:
        plt.plot(train_loss, lw=2)
        legend_label = ['train', 'test']
    elif train_loss.shape[1] == 3:
        plt.plot(train_loss[:,0], lw=2)
        plt.plot(train_loss[:,1], lw=2)
        plt.plot(train_loss[:,2], lw=2)
        legend_label = ['train', 'bce', 'jacob', 'test']
    else:
        raise ValueError('invalid train_loss shape')

    plt.plot(test_loss, lw=2)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(legend_label)
    plt.grid()
    plt.yscale('log')
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
    return


def save_latent_traversals(resdir, model, latent_size, img_shape, n_samples=48):

    dev = next(model.parameters()).device
    with torch.no_grad():

        # traverse each latent dimension
        for a in range(latent_size):

            sample = np.zeros((n_samples, latent_size), dtype=np.float32)
            sample[:,a] = np.linspace(-2, 2, n_samples)

            sample = torch.from_numpy(sample).to( dev )
            sample = model.decode(sample).cpu()

            save_image(sample.view(n_samples, 1, *img_shape),
                       pjoin(resdir, 'z' + str(a) + '.png'))

    return        


def save_samples(resdir, epoch, model, latent_size):

    dev = next(model.parameters()).device
    with torch.no_grad():

        # generate some random samples
        sample = torch.randn(64, latent_size).to(dev)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, *field_shape),
                   pjoin(resdir, 'samples/sample_%d.png' % epoch))

    return


def save_comparison(resdir, epoch, x, y, batch_size):

    raise NotImplementedError() # TODO

    n = min(x.shape[0], 8)
    comp_shp = (min(batch_size, x.shape[0]), 1, *x.shape[1:])
    comparison = torch.cat([x.view(*comp_shp)[:n],
                            y.view(*comp_shp)[:n]])
    save_image(comparison.cpu(),
               pjoin(resdir, 'reconstructions/reconstruction_%d.png' % epoch), nrow=n)
    return


