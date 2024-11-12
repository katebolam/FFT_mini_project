import numpy as np
from 2D_OpenMP_FFT import fft_2d_parallel

# Test the 2D FFT with random data
N = 512
data = np.random.random((N, N))
result = fft_2d_parallel(data)
print("2D FFT Result:", result)