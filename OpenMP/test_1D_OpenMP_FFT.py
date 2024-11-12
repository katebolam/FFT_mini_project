import numpy as np
from 1D_OpenMP_FFT import fft_1d_parallel

# Test the 1D FFT with random data
N = 1024
data = np.random.random(N)
result = fft_1d_parallel(data)
print("1D FFT Result:", result)
