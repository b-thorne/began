#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import h5py

import click
from IPython.core import ultratb

import healpy as hp 
import numpy as np
from astropy.io import fits
from began.visualization import mplot


# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)

@click.command()
@click.option('--input_path', 'input_path', required=True, 
                type=click.Path(exists=True), help='path to input data')
@click.option('--output_path', 'output_path', required=True,
                type=click.Path(), help='path to output file')
@click.option('-p', 'polarization', default=False)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('--seed', 'seed', default=12342, type=int)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(input_path: Path, output_path: Path, polarization: bool, seed: int, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info(
        """Dependencies:
            Input path: {:s} 
            Output directors: {:s}
    """.format(input_path, output_path))

    output_path = Path(output_path).absolute()


    xlabels = None
    ylabels = None
    logging.debug("Random seed: {:d}".format(seed))
    np.random.seed(seed)
    indices = np.sort(np.random.randint(0, 1033, 9))
    with h5py.File(input_path, 'r') as f:
        plot_data = f["norm_maps"][indices] 
    np.random.seed(None)
    titles = ["{:d}".format(i) for i in indices]
    extent = (-10, 10, -10, 10)
    title = "Random set of maps from training set"
    fig, axes = mplot(plot_data[..., 0], title=title, xlabels=xlabels, ylabels=ylabels, titles=titles, extent=extent)
    fig.savefig(output_path / "sample_of_training_maps_t.png", bbox_inches='tight')

    if polarization:
        title = "Random set of Q maps from training set"
        fig, axes = mplot(plot_data[..., 1], title=title, xlabels=xlabels, ylabels=ylabels, titles=titles, extent=extent)
        fig.savefig(output_path / "sample_of_training_maps_q.png", bbox_inches='tight')

        title = "Random set of U maps from training set"
        fig, axes = mplot(plot_data[..., 2], title=title, xlabels=xlabels, ylabels=ylabels, titles=titles, extent=extent)
        fig.savefig(output_path / "sample_of_training_maps_u.png", bbox_inches='tight')


if __name__ == '__main__':
    main()