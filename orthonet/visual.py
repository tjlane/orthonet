
from matplotlib import pyplot as plt
import numpy as np

def plot_loss_curves(train_loss, test_loss, save=False):
    """
    save : str
        path to save the image to, or False if no save desired
    """
    
    plt.figure()
    
    plt.plot(train_loss, lw=2)
    plt.plot(test_loss, lw=2)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'])
    plt.grid()
    plt.yscale('log')
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
    return
