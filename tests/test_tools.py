from began.tools import FlatCutter, spin2rot
import astropy.units as u
import healpy as hp
import numpy as np
import pytest 

@pytest.mark.parametrize('nside', [64, 128])
def testFlatCutter(nside):
    npix = hp.nside2npix(nside)

    # pix2ang returns colatitude and longitude in radians
    map1 = hp.pix2ang(nside, np.arange(npix))

    xlen = 10. * u.deg
    ylen = 10. * u.deg
    xres = 256
    yres = 256

    theta_rot = 20. * u.deg
    phi_rot = 10. * u.deg

    fc = FlatCutter(xlen, ylen, xres, yres)
    assert len(fc.xarr) == xres
    assert len(fc.yarr) == yres
    assert fc.co_lats.unit == u.rad
    assert fc.lons.unit == u.rad
    return


def test_spin2rot():
    q = np.random.randn(100)
    u = np.random.randn(100)
    phi = np.random.uniform(0, 2 * np.pi)
    rotq, rotu = spin2rot(q, u, phi)
    unrotq, unrotu = spin2rot(rotq, rotu, -phi)

    assert np.allclose(q, unrotq)
    assert np.allclose(u, unrotu)
    return