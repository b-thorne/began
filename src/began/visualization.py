import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from astropy.visualization import astropy_mpl_style
import math
import numpy as np
import matplotlib.units as units
import matplotlib.ticker as ticker

def mplot(ma, extent=None, title=None, titles=None, cbar_range=None, xlabels=None, ylabels=None):
    try:
        assert ma.ndim == 3
    except AssertionError:
        raise AssertionError("Number of dimensions must be three")

    nma = ma.shape[0]

    if cbar_range is not None:
        (vmin, vmax) = cbar_range
    else:
        vmin, vmax = None, None
    
    if nma <= 3:
        fig, axes = plt.subplots(1, nma, sharey=True, figsize=(4 * nma, 4))

    else:
        nplot = np.round(np.sqrt(nma)).astype(int)
        fig, axes = plt.subplots(nrows=nplot, ncols=nplot, sharex=True, sharey=True, figsize=(4 * nplot, 4 * nplot))
 
    plt.subplots_adjust(top=0.85, bottom=0.1)
    
    if title is not None:
        fig.suptitle(title, y=0.94)

    if xlabels is None:
        xlabels = [None for _ in axes.flatten()]
    if ylabels is None:
        ylabels = [None for _ in axes.flatten()]
    
    if titles is not None:
        for ax, _title in zip(axes.flatten(), titles):
            ax.set_title(_title)

    for i, (ax, xlabel, ylabel) in enumerate(zip(axes.flatten(), xlabels, ylabels)):
        im = plot(ma[i], extent=extent, xlabel=xlabel, ylabel=ylabel, ax=ax, ret_im=True, imshow_kw={'vmin': vmin, 'vmax': vmax})
        
    if cbar_range is not None:
        cb_ax = fig.add_axes([0.33, 0.03, 0.3, 0.05])
        fig.colorbar(im, cax=cb_ax, orientation='horizontal', cmap=plt.cm.coolwarm)
    return fig, axes


def plot(ma, fig=None, ax=None, extent=None, xlabel=None, ylabel=None, title=None, ret_im=False, cbar=False, imshow_kw={}):
    try:
        assert ma.ndim == 2
    except AssertionError:
        raise AssertionError("Number of dimensions must be two")
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if title is not None:
        ax.set_title(title)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.tick_params(axis="both", direction='in', length=4, width=1, colors='k')
    im = ax.imshow(ma, extent=extent, interpolation='nearest', origin='lower', cmap=plt.cm.coolwarm, **imshow_kw)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    
    if extent is not None:
        ax.set_xticklabels(list([r"${:d}^\circ$".format(int(t)) for t in ax.get_xticks()]))
        ax.set_yticklabels(list([r"${:d}^\circ$".format(int(t)) for t in ax.get_yticks()]))
    
    if not ret_im:
        return fig, ax
    return im