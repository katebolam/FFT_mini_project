# This script implements a 2D fast Fourier transform (FFT) for a process-parallelized case using the multiprocessing Python module,
# and stores and plots the execution times as a function of problem size (N)

import numpy as np
import sys
import os
from multiprocessing import Pool

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from timing import time_function
from plotting import plot_execution_times

# Define the problem sizes as powers of 2 for square matrices from 2^0 to 2^12
problem_sizes = [2**i for i in range(13)]  
execution_times = []  # To store mean execution times for each problem size

# Number of runs for averaging
num_runs = 1

# Function to time the 2D FFT for multiprocessing
def time_fft2(data):
    return time_function(np.fft.fft2, data)

# Main block to run the code with multiprocessing
if __name__ == "__main__":
    # Run the 2D FFT with multiprocessing for each problem size
    for size in problem_sizes:
        data = np.random.random((size, size))  # Generate random 2D data for FFT

        # Use a pool of worker processes to parallelize timing runs
        with Pool() as pool:
            # Run `time_fft2` across `num_runs` parallel executions
            times = pool.map(time_fft2, [data] * num_runs)

        mean_time = np.mean(times)  # Calculate mean execution time across runs
        execution_times.append(mean_time)

        print(f"2D (Multiprocessing) FFT Execution Time for {size}x{size}: {mean_time:.10f} seconds")

    # Plot the execution times with problem size N as x-axis
    plot_execution_times(
        problem_sizes,
        [execution_times],  
        labels=["Multiprocessing 2D FFT"], 
        title="2D FFT Multiprocessing Execution Time vs. Problem Size",
        xlabel="Problem Size (2^N x 2^N)",
        ylabel="Execution Time (s)"
    )
