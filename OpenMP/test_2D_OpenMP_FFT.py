import numpy as np
import sys
import os

# Add utils and OpenMP directories to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenMP')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from OpenMP_2D_FFT import fft_2d_parallel
from timing import time_function
from plotting import plot_execution_times

# Define the problem sizes as powers of 2 from 2^0 to 2^12
problem_sizes = [2**i for i in range(13)]
execution_times_2d = []  # To store mean execution times for each problem size

# Number of runs for averaging
num_runs = 10

# Test and measure execution time
for size in problem_sizes:
    data_2d = np.random.random((size, size))  # Generate random 2D data for FFT
    times_2d = [time_function(fft_2d_parallel, data_2d) for _ in range(num_runs)]
    mean_time_2d = np.mean(times_2d)
    execution_times_2d.append(mean_time_2d)
    print(f"2D (OpenMP) FFT Execution Time for N={size}x{size}: {mean_time_2d:.10f} seconds")

# Plot the execution times with problem size N as x-axis
plot_execution_times(
    problem_sizes,
    [execution_times_2d],
    labels=["OpenMP 2D FFT"],
    title="2D FFT OpenMP Execution Time vs. Problem Size",
    xlabel="Problem Size (NxN)",
    ylabel="Execution Time (s)"
)
