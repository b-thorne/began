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
from astropy.io import fits
from pathlib import Path
import h5py
import began
from began.logging import setup_vae_run_logging
from began.visualization import mplot, plot


# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)

@click.command()
@click.option('-c', '--cfg_path', 'cfg_path', required=True,
                type=click.Path(exists=True), help='path to config file of network')
@click.option('--model_path', 'model_path', required=True, 
                type=click.Path(exists=True), help='path to trained model')
@click.option('--output_path', 'output_path', required=True, 
                type=click.Path(exists=True) help='path to save sampled maps'
@click.option('--seed', 'seed', type=int, default=1234321)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(cfg_path: Path, train_path: Path, model_path: Path, plot_dir: Path, seed: int, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    main()