import numpy as np
cimport numpy
cimport cython


ctypedef numpy.int16_t DTYPE_t
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def os_cyth(numpy.ndarray[DTYPE_t, ndim=2] x, guard_len=0, noise_len=8, k=12, scale=1.0, axis=None):
    """Performs Ordered-Statistic CFAR (OS-CFAR) detection on the input array.

    Args:
        x (~numpy.ndarray): Noisy array to perform cfar on with log values
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        k (int): Ordered statistic rank to sample from.
        scale (float): Scaling factor.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.os_(signal, k=3, scale=1.1, guard_len=0, noise_len=3)
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([93, 59, 58, 58, 83, 59, 59, 58, 83, 83]), array([85, 54, 53, 53, 76, 54, 54, 53, 76, 76]))

    """
    cdef int j, n, cut_idx, run_idx

    if isinstance(x, list):
        x = np.array(x, dtype=np.uint32)

    if axis==None:
        assert 'Please insert axis!'
    noise_floor = np.zeros_like(x, dtype=np.float32)

    if axis==0:
        n=x.shape[1]
        run_idx = x.shape[0]
        for j in range(run_idx):
            # Initial CUT
            left_idx = list(np.arange(n - noise_len - guard_len - 1, n - guard_len - 1))
            right_idx = list(np.arange(guard_len, guard_len + noise_len))

            cut_idx = -1
            while cut_idx < (n - 1):
                cut_idx += 1

                left_idx.pop(0)
                left_idx.append((cut_idx - 1) % n)

                right_idx.pop(0)
                right_idx.append((cut_idx + guard_len + noise_len) % n)

                window = np.concatenate((x[j,left_idx], x[j,right_idx]))
                window.partition(k)
                noise_floor[j,cut_idx] = window[k]

    elif axis==1:
        n = x.shape[0]
        run_idx = x.shape[1]
        for j in range(run_idx):
            # Initial CUT
            left_idx = list(np.arange(n - noise_len - guard_len - 1, n - guard_len - 1))
            right_idx = list(np.arange(guard_len, guard_len + noise_len))

            cut_idx = -1
            while cut_idx < (n - 1):
                cut_idx += 1

                left_idx.pop(0)
                left_idx.append((cut_idx - 1) % n)

                right_idx.pop(0)
                right_idx.append((cut_idx + guard_len + noise_len) % n)

                window = np.concatenate((x[left_idx,j], x[right_idx,j]))
                window.partition(k)
                noise_floor[cut_idx,j] = window[k]

    return np.multiply(noise_floor, scale), noise_floor