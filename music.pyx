import numpy as np
cimport numpy
import numpy.linalg as LA

ctypedef numpy.complex128_t DTYPE_t
def music_cyth(numpy.ndarray[DTYPE_t, ndim=1] rx_chirps, int num_sources):
    cdef float ang_est_resolution = 0.2
    cdef int kk,jj, ang_est_range = 90, num_ant = 8


    R = np.cov(np.outer(rx_chirps, rx_chirps.T), rowvar=0)

    num_vec = (2 * ang_est_range / ang_est_resolution + 1)
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = complex(real, imag)

    _, v = LA.eigh(R)
    noise_subspace = v[:, :-num_sources]

    v = noise_subspace.T.conj() @ steering_vectors.T
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=0).real)

    return spectrum