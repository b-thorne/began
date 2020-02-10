#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import yaml

import click
from IPython.core import ultratb

import healpy as hp 
import numpy as np
import astropy.units as u
from astropy.io import fits
from began.tools import get_patch_centers, FlatCutter




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
@click.option('-p', 'polarization', default=False)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(cfg_path: Path, input_path: Path, output_path: Path, polarization: bool, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("""
    Configuration: {:s}
    Input {:s}
    Output: {:s}""".format(cfg_path, input_path, output_path))

    # read configuration file
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    gal_cut = config['tiling']['gal_cut'] * u.deg # latitudinal cut around galactic plane in degrees
    step_size = config['tiling']['step_size'] * u.deg # azimuthal step between patch centers in degrees
    ang_x = config['patch']['ang_x'] * u.deg # angular size of patch in degrees
    ang_y = config['patch']['ang_y'] * u.deg # angular size of patch in degrees
    xres = config['pixelization']['xres'] # pixelization in x dimension
    yres = config['pixelization']['yres'] # pixelization in y dimension

    # read map and infer nside
    if polarization:
        field = (0, 1, 2)
    else:
        field = 0
    input_map = hp.read_map(input_path, field=field, dtype=np.float64, verbose=False)
    logging.debug("Input map fits header: \n {:s}".format(repr(fits.open(input_path)[1].header)))
    
    logging.info(
        """Cutting map with tiling: 
        gal_cut: {:.01f}  
        step_size: {:.01f}""".format(gal_cut, step_size))

    centers = get_patch_centers(gal_cut, step_size)
    logging.debug("Number of patches: {:d}".format(len(centers)))

    logging.info(
        """Patch parameters: 
        Linear size in degrees: {:.01f} degrees by {:.01f} degrees
        Number of pixels: {:.01f} by {:.01f} 
        """.format(ang_x, ang_y, xres, yres))

    # cut out maps at each of the patch centers
    fc = FlatCutter(ang_x, ang_y, xres, yres)
    cut_maps = [fc.rotate_to_pole_and_interpolate(lon, lat, input_map) for (lon, lat) in centers]

    # rescale
    cut_maps = np.log(cut_maps)
    cut_maps = 2 * (cut_maps - cut_maps.min()) / (cut_maps.max() - cut_maps.min()) - 1.

    # save maps and add new axis at end corresponding to channel
    np.save(output_path, cut_maps)

if __name__ == '__main__':
    main()  