"""
# OpenMP_2D_FFT.pyx

This file contains the Cython implementation of a 2D Fast Fourier Transform (FFT) using OpenMP for shared memory parallelization.

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

# Import OpenMP API to dynamically set the number of threads
cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)

def fft_2d_parallel(np.ndarray[np.float64_t, ndim=2] data, int num_threads):
    """
    Perform a parallel 2D FFT with dynamically adjustable thread count.
    
    Parameters:
    - data: 2D input array of real numbers (Nx x Ny).
    - num_threads: Number of OpenMP threads to use.
    
    Returns:
    - output: Complex 2D FFT of the input data.
    """
    cdef int Nx = data.shape[0]
    cdef int Ny = data.shape[1]
    cdef np.ndarray[np.complex128_t, ndim=2] output = np.empty((Nx, Ny), dtype=np.complex128)
    cdef int kx, ky, n, m
    cdef double angle_real, angle_imag, cos_val, sin_val

    # Temporary arrays for real and imaginary components
    cdef np.ndarray[np.float64_t, ndim=2] temp_real = np.zeros((Nx, Ny), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] temp_imag = np.zeros((Nx, Ny), dtype=np.float64)

    # Set the number of threads dynamically
    omp_set_num_threads(num_threads)

    # Parallel loop for the Fourier transform
    with nogil:
        for kx in prange(Nx, schedule='dynamic'):
            for ky in range(Ny):
                for n in range(Nx):
                    for m in range(Ny):
                        angle_real = cos(-2.0 * M_PI * (kx * n / Nx + ky * m / Ny))
                        angle_imag = sin(-2.0 * M_PI * (kx * n / Nx + ky * m / Ny))
                        temp_real[kx, ky] += data[n, m] * angle_real
                        temp_imag[kx, ky] += data[n, m] * angle_imag

    # Combine the temporary real and imaginary parts
    for kx in range(Nx):
        for ky in range(Ny):
            output[kx, ky] = temp_real[kx, ky] + 1j * temp_imag[kx, ky]

    return output
