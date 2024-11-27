"""
# OpenMP_1D_FFT.pyx

This file contains the Cython implementation of a 1D Fast Fourier Transform (FFT) using OpenMP for shared memory parallelization.

Dependencies:
- Cython
- OpenMP
- NumPy

"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport exp, M_PI, cos, sin

# Import OpenMP API to control the number of threads
cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)

def fft_1d_parallel(np.ndarray[np.float64_t, ndim=1] data, int num_threads):
    """
    Perform a parallel 1D FFT with dynamically adjustable thread count.
    
    Parameters:
    - data: Input array of real numbers (1D).
    - num_threads: Number of OpenMP threads to use.
    
    Returns:
    - output: Complex 1D FFT of the input data.
    """
    cdef int N = data.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1] output = np.empty(N, dtype=np.complex128)
    cdef int k, n
    cdef double angle_real, angle_imag, cos_val, sin_val

    # Temporary arrays for real and imaginary components
    cdef np.ndarray[np.float64_t, ndim=1] temp_real = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] temp_imag = np.zeros(N, dtype=np.float64)

    # Set the number of threads dynamically
    omp_set_num_threads(num_threads)

    # Parallel loop for the Fourier transform
    with nogil:
        for k in prange(N, schedule='dynamic'):
            for n in range(N):
                angle_real = cos(-2.0 * M_PI * k * n / N)
                angle_imag = sin(-2.0 * M_PI * k * n / N)
                temp_real[k] += data[n] * angle_real
                temp_imag[k] += data[n] * angle_imag

    # Combine the temporary real and imaginary parts
    for k in range(N):
        output[k] = temp_real[k] + 1j * temp_imag[k]

    return output
