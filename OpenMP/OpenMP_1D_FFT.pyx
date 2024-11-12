import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport cos, sin, M_PI
cimport openmp

# 1D FFT function with OpenMP
def fft_1d_parallel(np.ndarray[double] data):
    cdef int N = data.shape[0]
    cdef int k, n
    cdef double complex_even, complex_odd, twiddle_real, twiddle_imag

    # Output arrays
    cdef np.ndarray[double] output_real = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double] output_imag = np.empty(N, dtype=np.float64)

    # Parallel loop with OpenMP for reduction
    with nogil:
        for k in prange(N, schedule='dynamic', num_threads=4):  
            complex_even = 0.0
            complex_odd = 0.0
            for n in range(N):
                twiddle_real = cos(-2.0 * M_PI * k * n / N)
                twiddle_imag = sin(-2.0 * M_PI * k * n / N)
                complex_even += data[n] * twiddle_real
                complex_odd += data[n] * twiddle_imag
            output_real[k] = complex_even
            output_imag[k] = complex_odd

    return output_real + 1j * output_imag
