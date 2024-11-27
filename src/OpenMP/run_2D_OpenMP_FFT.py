import argparse
import numpy as np
import sys
import os

# Add utils and OpenMP directories to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenMP')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from OpenMP_2D_FFT import fft_2d_parallel
from timing import time_function

# Argument parsing to accept threads as a command line argument
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, help='Number of threads for parallel processing')
    args = parser.parse_args()

    num_threads = args.threads  # Use the value passed from the SLURM script
    problem_sizes = [2**i for i in range(12, 29)]  # Define the problem sizes as powers of 2 from 2^12 to 2^28
    num_runs = 1  # Number of runs for averaging

    # Execution Time vs Problem Size
    for size in problem_sizes:
        data_2d = np.random.random((size, size))  # Generate random 2D data for FFT
        times_2d = [time_function(fft_2d_parallel, data_2d, num_threads) for _ in range(num_runs)]
        mean_time_2d = np.mean(times_2d)
        print(f"2D FFT Execution Time for N={size}x{size}, Threads={num_threads}: {mean_time_2d:.10f} seconds")

if __name__ == '__main__':
    main()

