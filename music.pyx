import numpy as np
cimport numpy
import numpy.linalg as LA

ctypedef numpy.complex128_t DTYPE_t
ctypedef numpy.complex64_t DTYPE_d
def music_cyth(numpy.ndarray[DTYPE_t, ndim=1] rx_chirps, numpy.ndarray[DTYPE_d, ndim=2] steering_vec, int num_sources):

    R = np.cov(np.outer(rx_chirps, rx_chirps.T), rowvar=0)
    _, v = LA.eigh(R)
    noise_subspace = v[:, :-num_sources]

    v = noise_subspace.T.conj() @ steering_vec.T
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=0).real)

    return spectrum