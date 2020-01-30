#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys

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
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('--seed', 'seed', default=12342, type=int)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(input_path: Path, output_path: Path, seed: int, log_level: int):
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
    train_data = np.load(input_path)

    xlabels = [None, None, r"$x$", r"$x$"]
    ylabels = [r"y", None, r"$y$", None]
    logging.debug("Random seed: {:d}".format(seed))
    np.random.seed(seed)
    indices = np.random.randint(0, 1033, 4)
    np.random.seed(None)
    plot_data = np.array([train_data[i] for i in indices])[..., 0]
    titles = ["{:d}".format(i) for i in indices]
    extent = (-10, 10, -10, 10)
    title = "Random set of maps from training set"
    fig, axes = mplot(plot_data, title=title, xlabels=xlabels, ylabels=ylabels, titles=titles, extent=extent)
    fig.savefig(output_path / "sample_of_training_maps.pdf")

if __name__ == '__main__':
    main()