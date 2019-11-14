import os
import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt

def get_patches(npatches):
    """ Function to split a given HEALPix map into flat sky projections of
    a given dimensionality. Given a number of patches, this function will
    determine the positions and sizes of the corresponding pathces.

    This function implicitly assumes that the flat sky approximation is
    appropriate, and therefore, that the angles subtended by each patch
    are small.

    Paramemters
    -----------
    inmap: ndarray
        HEALPix map to be split.
    npathes: int
        Number of patches into which `inmap` should be split.
    res: int (optional, default=900)
        res gives the

    Returns
    -------
    ndarray
        Numpy array of shape (npatches, xres, yres).
    """
    try:
        assert is_square(npatches)
    except AssertionError:
        raise AssertionError("npatches must be a square number.")
    try:
        assert isinstance(npatches, int)
    except AssertionError:
        raise AssertionError("npatches must be an integer.")
    nslices = int(np.sqrt(npatches))
    # determine longitude ranges in HEALPix convention.
    lonras = get_slice_bounds(nslices, 0, 360)
    # determine latitude ranges in HEALPix convention.
    latras = get_slice_bounds(nslices, - 90, 90)
    # combine lonra and latra to get definitions of all patches
    # each elemnt of this list is a dictionary with lonra and latra
    # keys.
    patches = []
    for lonra in lonras:
        for latra in latras:
            patches.append({'lonra': lonra, 'latra': latra})
    return patches

def get_slice_bounds(nslices, lo, hi):
    """ Function to return the lower and upper edges of nslices slices in a range
    given by the elements of bounds.

    Parameters
    ----------
    nslices: int
        Number of slices to split the range into.
    lo, hi: float
        The lower and upper bound of the range to be considered.

    Returns
    -------
    generator
        Generator for a list that has nslices elements, each of which is a tuple of two
        numbers demarking the lower and upper bounds of each slice.
    """
    # split range given by bounds[0] to bounds[1] into nslices. bounds[1] is not
    # included, so lo_bnds has only the lower edge of each slice.
    lo_bnds, step = np.linspace(lo, hi, nslices, endpoint=False, retstep=True)
    return [[lo_bnd, lo_bnd + step] for lo_bnd in lo_bnds]

def is_square(pos_int):
    """ Function to check if a given number is a perfect square.

    Parameters
    ----------
    pos_int: int
        A given positive integer to be tested.

    Returns
    -------
    bool
        True if pos_int is squre, False otherwise.
    """
    x = pos_int // 2
    seen = set([x])
    while x * x != pos_int:
        x = (x + (pos_int // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True

def apply_normalization():
    """ Function to rescale a given map to the interval (-1, 1) required
    by the GAN algorithm. 

    Firstly, the map is 
    """
    return 

if __name__ == '__main__':
    # Resolution parameter. This dictates the number of pixels (RES ** 2) in
    # the patch maps. The resolution of the 545 GHz map is 5 arcminutes. We
    # want at least a few pixels per beam diameter in order to properly sample
    # the beam. This sets a lower limit on RES. After splitting the sky into
    # NPATCHES patches, each region will have size 
    RES = 2 ** 8
    NPATCHES = 2 ** 10
    FILE = "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits"
    DATA_DIR = os.path.realpath("/home/bthorne/projects/gan/began/data/raw")
    PROC_DIR = os.path.realpath("/home/bthorne/projects/gan/began/data/processed")
    MAP_BASE = os.path.splitext(FILE)[0]
    HFILE_NAME = "".join([MAP_BASE, ".hdf5"])
    HFILE_PATH = os.path.join(PROC_DIR, HFILE_NAME)

    # Read the given map in FILE
    INMAP = hp.read_map(os.path.join(DATA_DIR, FILE), verbose=False)
    # Get the longitude and latitude ranges for each patch
    PATCHES = get_patches(NPATCHES)
    # For each patch make a Cartesian projection
    CART = np.empty((NPATCHES, RES, RES))
    LONS = np.empty((NPATCHES, 2))
    LATS = np.empty((NPATCHES, 2))
    for i, patch in enumerate(PATCHES):
        CART[i] = hp.cartview(INMAP, title='Cartview {:d}'.format(i), return_projected_map=True, xsize=RES, ysize=RES, **patch)
        plt.close('all')
        LONS[i] = patch['lonra']
        LATS[i] = patch['latra']

    # Write to file
    with h5py.File(HFILE_PATH, 'w') as f:
        intensity = f.create_group("Intensity")
        planck = intensity.create_group("Planck")
        # Record the original map
        orig = planck.create_dataset("Original", data=INMAP)
        orig.attrs['NSIDE'] = hp.get_nside(INMAP)
        orig.attrs['FILE_NAME'] = FILE
        # Save the renormalized and cutout maps
        renorm = planck.create_dataset("Renormalized", data=CART)
        renorm.attrs['NPATCHES'] = NPATCHES
        renorm.attrs['FILE'] = FILE
        renorm.attrs['RES'] = RES
        renorm.attrs['LONRAS_HPIX'] = LONS
        renorm.attrs['LATRAS_HPIX'] = LATS


                