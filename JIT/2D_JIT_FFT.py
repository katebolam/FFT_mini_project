import sys
import os
from numba import jit, prange
import math
import time

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Numba-compatible random 2D data generator
@jit(nopython=True)
def generate_random_data_2d(size):
    """
    Generate random 2D data using a Numba-supported random number generator.
    """
    data = [[0.0] * size for _ in range(size)] 
    for i in range(size):
        for j in range(size):
            data[i][j] = math.sin(i * j) + math.cos(i + j)  # Deterministic pseudo-random values
    return data

# Numba-compatible simplified 2D FFT implementation
@jit(nopython=True)
def simplified_fft_2d(data):
    """
    Simplified 2D FFT computation using a manual approach, i.e., bypassing np.fft function.
    """
    N = len(data)
    result_real = [[0.0] * N for _ in range(N)]  # Real part of the result
    result_imag = [[0.0] * N for _ in range(N)]  # Imaginary part of the result

    # Perform the FFT on rows
    for i in range(N):
        for k in range(N):
            sum_real = 0.0
            sum_imag = 0.0
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                sum_real += data[i][n] * math.cos(angle)
                sum_imag += data[i][n] * math.sin(angle)
            result_real[i][k] = sum_real
            result_imag[i][k] = sum_imag

    # Perform the FFT on columns
    final_real = [[0.0] * N for _ in range(N)]
    final_imag = [[0.0] * N for _ in range(N)]
    for k in range(N):
        for j in range(N):
            sum_real = 0.0
            sum_imag = 0.0
            for n in range(N):
                angle = -2 * math.pi * j * n / N
                sum_real += result_real[n][k] * math.cos(angle) - result_imag[n][k] * math.sin(angle)
                sum_imag += result_real[n][k] * math.sin(angle) + result_imag[n][k] * math.cos(angle)
            final_real[j][k] = sum_real
            final_imag[j][k] = sum_imag

    return final_real, final_imag

# JIT-decorated function with parallelization for 2D FFT
@jit(nopython=True, parallel=True)
def compute_fft_2d_parallel(num_runs, size):
    """
    Perform 2D FFT computations in parallel across multiple runs.
    """
    for _ in prange(num_runs):  # Parallel loop
        data = generate_random_data_2d(size)  # Generate random 2D data
        simplified_fft_2d(data)  # Compute 2D FFT

# Main function for benchmarking and printing execution times
if __name__ == "__main__":
    # Define problem sizes (N x N) and other parameters
    problem_sizes = [2**i for i in range(13)] 
    num_runs = 1  # Number of iterations per problem size
    execution_times = []

    # Benchmark each problem size
    for size in problem_sizes:
        start = time.perf_counter()
        compute_fft_2d_parallel(num_runs, size)
        end = time.perf_counter()
        avg_time = (end - start) / num_runs
        execution_times.append(avg_time)
        
        # Print execution time for this iteration
        print(f"Problem Size {size}x{size}: Execution Time = {avg_time:.10f} seconds")
