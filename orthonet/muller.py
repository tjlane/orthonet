
import numpy as np
from matplotlib import pyplot as plt

def central_diff(f, x, epsilon=1e-8):
    """
    f : function
    x : point (should be 1-d)
    """
    
    D = len(x) # dimensionality
    
    grad = np.zeros(D)
    
    for d in range(D):
        xp = x.copy()
        xm = x.copy()
        xp[d] += epsilon
        xm[d] -= epsilon

        grad[d] = (f(*xp) - f(*xm)) / (2.0 * epsilon)
    
    return grad


class MullerForce:

    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        return
    
    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + \
                cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) + cls.cc[j] * (y - cls.YY[j])**2)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-2.0, maxx=1.5, miny=-1.5, maxy=2.5, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt.subplot(111)
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)
        return ax

        
    @classmethod
    def sample_trajectory(cls, length_t, initial_xy=None, beta=1e-3, D=0.01):
        """
        dx / dt = - beta * grad{V(x)} + srqt{2D} * R(t)
        """
        
        traj = np.zeros((length_t, 2))
        
        sq2D = np.sqrt(2.0 * D)
        
        if initial_xy is None:
            pass # zero zero for now
        else:
            traj[0] = initial_xy
            
        for t in range(1,length_t):
            gradV = central_diff(cls.potential, traj[t-1])
            traj[t] = traj[t-1] - beta * gradV + sq2D * np.random.randn(2)
        
        return traj


if __name__ == '__main__':
    mf = MullerForce()
    test_traj = mf.sample_trajectory(10000)
    ax = plt.gca()
    mf.plot(ax=ax)
    ax.scatter(test_traj[::100,0], test_traj[::100,1], color='k')


