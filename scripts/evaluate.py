#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
from pathlib import Path
import sys
import time

import click
from IPython.core import ultratb

import tensorflow as tf
import healpy as hp 
import numpy as np
import pymaster as nmt
from astropy.io import fits
from pathlib import Path
import h5py
import began
from began.visualization import mplot, plot
from began import stats

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)

@click.command()
@click.option('--input_path', 'input_path', required=True,
                type=click.Path(exists=True), help='path to input file from which to load maps')
@click.option('--output_dir', 'output_dir', type=click.Path(exists=True), required=True)
@click.option('--plot_dir', 'plot_dir', type=click.Path(exists=True), required=True)
@click.option('--batch_size', 'batch_size', type=int, default=-1, help='Size of batch to calculate')
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(input_path: Path, output_dir: Path, plot_dir: Path, batch_size: int, seed: int, log_level: int):
    # initialize random seed in numpy
    np.random.seed(seed)
    # initialize random seed in tensorflow
    tf.random.set_seed(seed)
    
    plot_dir = Path(plot_dir).absolute()

    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # get samples from the model
    _logger.info("""
    Data being loaded: {:s}
    Output directory: {:s}
    Batch size used: {:d}
    """.format(input_path, output_dir, batch_size))

    ang_x = 20.
    ang_y = 20.
    npix_x = 256
    npix_y = 256
    aposize = 1.
    _logger.info("""Creating mask with the specifications:
    x resolution: {:d} pixels
    y resolution: {:d} pixels
    x linear size: {:.03f} degrees
    y linear size: {:.03f} degrees
    apodization scale: {:.03f} degrees
    """.format(npix_x, npix_y, ang_x, ang_y, aposize))

    # read in data
    map_batch = np.load(input_path)[:batch_size, :, :, 0]
    assert map_batch.ndim == 3
    _logger.debug(repr(map_batch.shape))

    # calculate one-dimensional histogram

    # create flat mask with taper at edges
    mask = stats.build_flat_mask(npix_x, npix_y, ang_x * np.pi / 180., ang_y * np.pi / 180., aposize)
    fig, ax = plot(mask, xlabel="x", ylabel="y", title="Apodized mask", extent=(-10, 10, -10, 10))
    fig.savefig(plot_dir / "apodized_mask.pdf", bbox_inches='tight')

    # evaluate the power spectrum of the sampled maps
    nmtbin = stats.dimensions_to_nmtbin(npix_x, npix_y, ang_x * np.pi / 180., ang_y * np.pi / 180.)
    auto_spectra = stats.batch_00_autospectrum(map_batch, ang_x, ang_y, mask, nmtbin)
    print(auto_spectra.shape)
    # calculate the Frechet distance from the distribution of power spectra in the training set

    # save summary statistics in hdf5 file



if __name__ == '__main__':
    main()