import numpy as np
import sys
import os


# Add utils and OpenMP directories to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenMP')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from OpenMP_1D_FFT import fft_1d_parallel
from timing import time_function

# Define the problem sizes as powers of 2 from 2^0 to 2^12
problem_sizes = [2**i for i in range(13)]
num_runs = 10  # Number of runs for averaging
fixed_threads = 4  # Fixed number of threads for problem size plot

# Execution Time vs Problem Size (Fixed Threads)
for size in problem_sizes:
    data_1d = np.random.random(size)  # Generate random 1D data for FFT
    times_1d = [time_function(fft_1d_parallel, data_1d, fixed_threads) for _ in range(num_runs)]
    mean_time_1d = np.mean(times_1d)
    print(f"1D FFT Execution Time for N={size}, Threads={fixed_threads}: {mean_time_1d:.10f} seconds")

# Execution Time vs Number of Threads (Fixed Problem Size)
fixed_problem_size = 2**10  # Fix the problem size
data_fixed_size = np.random.random(fixed_problem_size)  # Generate random data for fixed size
threads = range(1, 9)  # Number of threads from 1 to 8

for thread_count in threads:
    times_threads = [time_function(fft_1d_parallel, data_fixed_size, thread_count) for _ in range(num_runs)]
    mean_time_threads = np.mean(times_threads)
    print(f"1D FFT Execution Time for Fixed Problem Size={fixed_problem_size}, Threads={thread_count}: {mean_time_threads:.10f} seconds")