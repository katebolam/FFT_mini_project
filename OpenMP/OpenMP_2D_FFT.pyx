import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport cos, sin, M_PI
cimport openmp

# 2D FFT function with OpenMP
def fft_2d_parallel(np.ndarray[double] data):
    cdef int N, M, i, j, n
    N, M = data.shape  # Get dimensions of the 2D array
    cdef double complex_even, complex_odd, twiddle_real, twiddle_imag

    # Output arrays for real and imaginary parts
    cdef np.ndarray[double] output_real = np.empty((N, M), dtype=np.float64)
    cdef np.ndarray[double] output_imag = np.empty((N, M), dtype=np.float64)

    # Parallelize row-wise FFT
    with nogil:
        for i in prange(N, schedule='dynamic', num_threads=4): 
            for j in range(M):
                complex_even = 0.0
                complex_odd = 0.0
                for n in range(N):
                    twiddle_real = cos(-2.0 * M_PI * i * n / N)
                    twiddle_imag = sin(-2.0 * M_PI * i * n / N)
                    complex_even += data[n, j] * twiddle_real
                    complex_odd += data[n, j] * twiddle_imag
                output_real[i, j] = complex_even
                output_imag[i, j] = complex_odd

    # Parallelize column-wise FFT
    with nogil:
        for j in prange(M, schedule='dynamic', num_threads=4): 
            for i in range(N):
                complex_even = 0.0
                complex_odd = 0.0
                for n in range(M):
                    twiddle_real = cos(-2.0 * M_PI * j * n / M)
                    twiddle_imag = sin(-2.0 * M_PI * j * n / M)
                    complex_even += data[i, n] * twiddle_real
                    complex_odd += data[i, n] * twiddle_imag
                output_real[i, j] = complex_even
                output_imag[i, j] = complex_odd

    # Combine real and imaginary parts
    return output_real + 1j * output_imag
