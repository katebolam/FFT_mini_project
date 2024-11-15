# This script implements a 1D fast-fourier transform (FFT) using Numba's Just-In-Time (JIT) compiler,
# and stores and plots the execution times as a function of problem size (N).

import sys
import os
from numba import jit, prange
import math
import time
import matplotlib.pyplot as plt

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from timing import time_function
from plotting import plot_execution_times

# Numba-compatible random number generator
@jit(nopython=True)
def generate_random_data(size):
    """
    Generate random data using a Numba-supported random number generator.
    """
    data = [0.0] * size  
    for i in range(size):
        data[i] = math.sin(i) + math.cos(i)  # Deterministic pseudo-random values, i.e. seem 'random', but are fixed and repeatable for each i
    return data

# Numba-compatible simplified FFT implementation
@jit(nopython=True)
def simplified_fft(data):
    """
    Simplified FFT computation using a manual approach, i.e. bypassing np.fft function, as this is not supported by
    Numba in 'nopython=True' mode.
    """
    N = len(data)
    result_real = [0.0] * N  # Real part
    result_imag = [0.0] * N  # Imaginary part

    for k in range(N):
        sum_real = 0.0
        sum_imag = 0.0
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            sum_real += data[n] * math.cos(angle)
            sum_imag += data[n] * math.sin(angle)
        result_real[k] = sum_real
        result_imag[k] = sum_imag

    return result_real, result_imag

# JIT-decorated function with parallelization
@jit(nopython=True, parallel=True)
def compute_fft_parallel(num_runs, size):
    """
    Perform FFT computations in parallel across multiple runs.
    """
    for _ in prange(num_runs):  # Parallel loop
        data = generate_random_data(size)  # Generate random 1D data
        simplified_fft(data)  # Compute FFT

# Main function for benchmarking and plotting
if __name__ == "__main__":
    # Define problem sizes and other parameters
    problem_sizes = [2**i for i in range(13)]  # 2^0 to 2^12
    num_runs = 10  # Number of iterations per problem size
    execution_times = []

    for size in problem_sizes:
        start = time.perf_counter()
        compute_fft_parallel(num_runs, size)
        end = time.perf_counter()
        avg_time = (end - start) / num_runs
        execution_times.append(avg_time)
        
        # Print execution time for this iteration
        print(f"Problem Size 2^{int(math.log2(size))}: Execution Time = {avg_time:.10f} seconds")

    # Plot execution times
    plot_execution_times(
        problem_sizes,
        [execution_times],
        labels=["Parallel 1D FFT (Numba)"],
        title="Execution Time vs. Problem Size for Parallel 1D FFT",
        xlabel="Problem Size (2^N)",
        ylabel="Execution Time (seconds)"
    )
