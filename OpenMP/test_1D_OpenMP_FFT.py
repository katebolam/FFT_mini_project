import numpy as np
import sys
import os

# Add utils and OpenMP directories to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenMP')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from OpenMP_1D_FFT import fft_1d_parallel
from timing import time_function
from plotting import plot_execution_times

# Define the problem sizes as powers of 2 from 2^0 to 2^12
problem_sizes = [2**i for i in range(13)]
execution_times_1d = []  # To store mean execution times for each problem size

# Number of runs for averaging
num_runs = 10

# Test and measure execution time
for size in problem_sizes:
    data_1d = np.random.random(size)  # Generate random 1D data for FFT
    times_1d = [time_function(fft_1d_parallel, data_1d) for _ in range(num_runs)]
    mean_time_1d = np.mean(times_1d)
    execution_times_1d.append(mean_time_1d)
    print(f"1D (OpenMP) FFT Execution Time for N={size}: {mean_time_1d:.10f} seconds")

# Plot the execution times with problem size N as x-axis
plot_execution_times(
    problem_sizes,
    [execution_times_1d],  
    labels=["OpenMP 1D FFT"], 
    title="1D FFT OpenMP Execution Time vs. Problem Size",
    xlabel="Problem Size (2^N)",
    ylabel="Execution Time (s)"
)
