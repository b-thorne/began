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
from astropy.io import fits
import began

def get_patch_centers(gal_cut, step_size):
    ""
    southern_lat_range = list(np.arange(-90, -gal_cut, step_size))
    northern_lat_range = list(np.arange(gal_cut + step_size, 90, step_size))
    lat_range = list(np.concatenate((southern_lat_range, northern_lat_range)))

    centers = []
    for t in lat_range:
        step = step_size / np.cos(t * np.pi / 180.)
        for i in np.arange(0, 360, step):
            centers.append((i, t))
    return centers


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
    logging.info("""
    Configuration: {:s}
    Input {:s}
    Output: {:s}""".format(cfg_path, input_path, output_path))

    # read configuration file
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    gal_cut = config['tiling']['gal_cut'] # latitudinal cut around galactic plane in degrees
    step_size = config['tiling']['step_size'] # azimuthal step between patch centers in degrees
    ang_x = config['patch']['ang_x'] # angular size of patch in degrees
    ang_y = config['patch']['ang_y'] # angular size of patch in degrees
    xres = config['pixelization']['xres'] # pixelization in x dimension
    yres = config['pixelization']['yres'] # pixelization in y dimension

    # read map and infer nside
    input_map = hp.read_map(input_path, dtype=np.float64, verbose=False)
    nside = hp.get_nside(input_map)
    logging.debug("Nside inferred from input map: {:d}".format(nside))
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
    fc = began.FlatCutter(ang_x, ang_y, xres, yres)
    cut_maps = [fc.rotate_and_interpolate(center, input_map) for center in centers]
    
    # rescale
    cut_maps = np.log(cut_maps)  
    cut_maps = 2 * (cut_maps - cut_maps.min()) / (cut_maps.max() - cut_maps.min()) - 1.

    # save maps and add new axis at end corresponding to channel
    np.save(output_path, cut_maps[..., None])

if __name__ == '__main__':
    main()  