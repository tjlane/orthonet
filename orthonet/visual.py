
from matplotlib import pyplot as plt
import numpy as np

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
