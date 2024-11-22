import argparse
import numpy as np
import sys
import os
from multiprocessing import Pool

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from timing import time_function
from plotting import plot_execution_times

# Define the problem sizes as powers of 2 for square matrices from 2^0 to 2^10
problem_sizes = [2**i for i in range(11)]
execution_times = []  # To store mean execution times for each problem size

# Number of runs for averaging
num_runs = 1

# Function to time the 2D FFT for multiprocessing
def time_fft2(data):
    return time_function(np.fft.fft2, data)

# Main block to run the code with argparse
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="2D FFT using Multiprocessing")
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of processes to use (default: 4)"
    )
    args = parser.parse_args()

    num_processes = args.processes  # Get number of processes from command-line arguments

    # Run the 2D FFT with multiprocessing for each problem size
    for size in problem_sizes:
        data = np.random.random((size, size))  # Generate random 2D data for FFT

        # Use a pool of worker processes to parallelize timing runs
        with Pool(processes=num_processes) as pool:  # Use argument for number of processes
            # Run `time_fft2` across `num_runs` parallel executions
            times = pool.map(time_fft2, [data] * num_runs)

        mean_time = np.mean(times)  # Calculate mean execution time across runs
        execution_times.append(mean_time)

        print(f"2D (Multiprocessing) FFT Execution Time for {size}x{size}, Processes={num_processes}: {mean_time:.10f} seconds")
