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
import began

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)

@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--input_path', 'input_path', required=True, 
                type=click.Path(exists=True), help='path to input data')
@click.option('--output_path', 'output_path', required=True,
                type=click.Path(), help='path to output file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(cfg_path: Path, input_path: Path, output_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Dependencies: \n \t Configuration: {:s} \n \t Input {:s} \n \t Output: {:s}".format(cfg_path, input_path, output_path))

    # read map
    input_map = hp.read_map(input_path, dtype=np.float64, verbose=False)
    nside = hp.get_nside(input_map)
    logging.debug("Nside inferred from input map: {:d}".format(nside))
    logging.debug("Input map fits header: \n {:s}".format(repr(fits.open(input_path)[1].header)))

    # give values for the 
    RES = 256
    GAL_CUT = 16
    STEP_SIZE = 4
    logging.info(
        """Cutting map with parameters: 
        RES: {:d}  
        GAL_CUT: {:d}  
        STEP_SIZE: {:d}""".format(RES, GAL_CUT, STEP_SIZE))

    southern_lat_range = list(np.arange(-90, -GAL_CUT, STEP_SIZE))
    northern_lat_range = list(np.arange(GAL_CUT + STEP_SIZE, 90, STEP_SIZE))
    lat_range = list(np.concatenate((southern_lat_range, northern_lat_range)))

    centers = []
    for t in lat_range:
        step = STEP_SIZE / np.cos(t * np.pi / 180.)
        for i in np.arange(0, 360, step):
            centers.append((i, t))

    logging.debug("Number of patches: {:d}".format(len(centers)))

    xlen = 10
    ylen = 10
    xres = 256
    yres = 256

    logging.info(
        """Patch parameters: 
        Linear size in degrees: {:d} degrees by {:d} degrees
        Number of pixels: {:d} by {:d} 
        """.format(xlen, ylen, xres, yres))

    fc = began.FlatCutter(xlen, ylen, xres, yres)

    cut_maps = [fc.rotate_and_interpolate(center, input_map) for center in centers]
    cut_maps = np.array(cut_maps)[..., None]

    cut_maps = np.log(cut_maps)  
    cut_maps = 2 * (cut_maps - cut_maps.min()) / (cut_maps.max() - cut_maps.min()) - 1.

    np.save(output_path, cut_maps)

if __name__ == '__main__':
    main()  