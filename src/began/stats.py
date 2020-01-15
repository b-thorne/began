import numpy as np
from scipy.linalg import sqrtm
import pymaster as nmt
import matplotlib.pyplot as plt

class StatsError(Exception):
    """ General python exception-derived object to raise erros within the 
    `stats` submodule.
    """


def frechet_distance(mean_1, mean_2, cov_1, cov_2):
    """ Function to calculate the Frechet distance between two curves.

    Parameters
    ----------
    mean_1, mean_2: ndarray
        Mean vector for the first and second lines.
    cov_1, cov_2: ndarray
        Covariance between points in the first line, and between points
        in the second line, respectively.
    
    Returns
    -------
    float
        Scalar `d_f`, the Frechet distance.
    """
    _assertNdSquareness(cov_1, cov_2)
    _assertSameLastTwoDims(cov_1, cov_2)
    _assertEqualLength(mean_1, mean_2)
    _assertFinite(mean_1, mean_2, cov_1, cov_2)
    trace_arg = cov_1 + cov_2  - 2. * square_root_mat(np.dot(cov_1, cov_2))
    return np.abs(mean_1 - mean_2) ** 2 + np.trace(trace_arg)


def square_root_mat(mat):
    """ A positive semi-definite matrix, B, has precisely one square root, A,
    such that AA=B. This function assumes a positive semi-definite matrix,
    and calculates, B, and returns A.

    Assumes that mat is square, rank 2, finite, and positive semi-definite.
    Matrices violating any of these assumptions will raise a `StatsError`.

    Under these conditions we find the eigenvalues, `v`, and matrix of
    column eigenvalues, `w`. The square root of B is then calculated as

    B = w . diag(sqrt(v)) . w.T

    Eigenvalues have 1e-10 * max(v) added to them as in some cases the limited
    numerical accuracy allos negative values for evalues close to zero.

    Parameters
    ----------
    mat: ndarray
        Array in which last two dimensions are square. Will return the square
        root of the final two dimensions.

    Returns
    -------
    ndarray
        Array containing the square roots of the matrix, `mat`. 


    Example
    -------
    > x = np.random.randn((10, 10))
    > B = np.dot(x, x.transpose())
    > A = square_root_mat(arr_psd)
    > print(B)
    > print(A)
    """
    _assertFinite(mat)
    _assertRankTwo(mat)
    _assertNdSquareness(mat)
    _assertPositiveSemiDefinite(mat)
    # take eigenvalues and eigenvectors of positive semidefinite matrix
    vs, w = np.linalg.eigh(mat)
    vs[np.where(vs<0)] = 0
    sqrt = np.dot(np.dot(w, np.diag(np.sqrt(vs))), w.transpose())
    return sqrt



def pixel_intensity_histogram(arr, nbins, hist_range=None, normed=False):
    """ Function to calculate the pixel intensity histogram for a set of
    maps.

    Assumes a set of maps in the shape (N, M, M, 1) as a result from the
    GAN.

    Parameters
    ----------
    maps: ndarray
        Array of shape (N, M, M, 1).
    nbins: int
        Number of bins.
    range: list(float) (optional, default=None)
        Pair of float, if provided will specify the range over which
        to define the bins.
    Returns
    -------

    ndarray
        Array containing binned histograms of pixel intensity in shape
        (N, bin_size)
    """
    _assertFinite(arr)
    _assertOneD(arr)
    return np.histogram(arr, nbins, range=hist_range, normed=normed)


def batch_00_autospectrum(ma, spanx, spany, mask, nmtbin, wsp):
    """ This is a function to calculate a batch estimate of the power
    spectrum for a set of flat-sky maps. It uses NaMaster to do all
    of the calculations of power spectra and mode coupling.

    This can be called with an existing `pymaster.NmtWorkspace`
    object, such that the mode coupling does not have to be calculated
    for each map.

    Parameters
    ----------
    ma: ndarray
        Numpy array with three dimensions corresponding to (NBATCH,
        XRES, YRES).
    spanx, spany: float
        Dimensions of the map in radians
    mask: ndarray
        Mask of same shape as map.
    wsp: `pymaster.NmtWorkspace` object
        Instance of a pymaster.NmtWorkspace object containing the
        mode coupling matrix.

    Returns
    -------
    ndarray
        Numpy array containing the power spectra, of shape (NBATCH
        N_BANDS).
    """
    _assertNdim(ma, 3)
    return np.array(list(map(map_to_00_autospectrum(ma, spanx, spany, mask, nmtbin, wsp))))


def map_to_00_autospectrum(ma, spanx, spany, mask, nmtbin, wsp):
    """ Function to calculate the auto spectrum of a given map, given
    the angular range it spans, a mask, a binning scheme, and a
    pre-computed mode coupling matrix.

    Parameters
    ----------
    ma: ndarray
        A two-dimensional numpy array.
    spanx, spany: float
        The angular distance spanned in the x and y directions respectively.
    mask: ndarray
        A two-dimensional numpy array containing the mask to be applied
        to the `ma` map.
    nmtbin: pymaster.NmtBinFlat object
        Instance of the Namaster flat sky binning object.
    wsp: pymaster.NmtWorkspaceFlat
        Instance of the Namaster flat sky workspace object with a precomputed
        mode coupling matrix.

    Returns
    -------
    ndarray
        The auto spectrum of the input map `ma`.
    """
    f0 = nmt.NmtFieldFlat(spanx, spany, mask, [ma])
    return wsp.decouple_cell(nmt.compute_coupled_cell_flat(f0, f0, nmtbin))


def dimensions_to_nmtbin(npix_x, npix_y, spanx, spany):
    """ Function to create a pymaster.NmtBinFlat object from the
    dimensions of a given map.

    Parameters
    ----------
    npix_x, npix_y: float
        Number of pixels along the x and y dimensions respectively.
    spanx, spanx: float
        The physical angular range spanned by each dimension in radians.
    """
    l0_bins = np.arange(npix_x / 8.) * 8 * np.pi / spanx
    lf_bins = (np.arange(npix_x / 8.) + 1.) * 8. * np.pi / spany
    return nmt.NmtBinFlat(l0_bins, lf_bins)


""" The following are functions used to check inputs for the above functions.
Some subset of these were copied directly from `numpy`, and the others are 
novel.
"""
def _assertOneD(arr):
    if arr.ndim != 1:
        raise StatsError("Array must be one dimensional")


def _assertRankTwo(mat):
    if mat.ndim != 2:
        raise StatsError("Matrix must be rank two to calculate square root")


def _assertNdim(ma, ndim):
    if ma.ndim != ndim:
        raise StatsError("Array must have {:d} dimensions.".format(ndim))


def _assertPositiveSemiDefinite(mat):
    v = np.linalg.eigvals(mat)
    # normalize by the largest magnitude eigenvalue and add a small amount
    # to account for numerical error in computing zero eigenvalues for 
    # uniform matrices.
    v /= np.max(np.abs(v))
    v += 1e-15
    if not all(v >= 0):
        raise StatsError("mat must be positive semidefinite. At least one eigenvalue is negative")


def _assertNdSquareness(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise StatsError('Last 2 dimensions of the array must be square')


def _assertEqualLength(*arrays):
    for arr in arrays:
        if arr.ndim != 1:
            raise StatsError("can not compare lengths of multidimensional arrays")
    if (len(set([arr.shape[0] for arr in arrays])) != 1):
        raise StatsError("mean_1 and mean_2 must be same length")


def _assertSameLastTwoDims(*arrays):
    for arr in arrays:
        if arr.ndim < 2:
            raise StatsError("must be at least rank 2 to compare last two dimensions")
    if (len(set([arr.shape[-2:] for arr in arrays])) != 1):
        raise StatsError("cov_1 and cov_2 must have same shape in last two dimensions")


def _assertFinite(*arrays):
    for a in arrays:
        if not (np.isfinite(a).all()):
            raise StatsError("Array must not contain infs or NaNs")