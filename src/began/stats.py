import numpy as np
from scipy.linalg import sqrtm
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