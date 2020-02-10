from began.tools import FlatCutter, spin2rot, get_patch_centers
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


def test_get_patch_centers():
    def is_in_plane(lat):
        return ((lat>-gal_cut-step_size) and (lat < gal_cut + step_size))
    gal_cut = 15. * u.deg
    step_size = 4. * u.deg
    centers = get_patch_centers(gal_cut, step_size)
    for lon, lat in centers:
        assert lon.unit == u.deg
        assert lat.unit == u.deg

        assert not is_in_plane(lat).any()
        assert is_in_plane(10. * u.deg)

    return