# this script implements a 1D fast-fourier transform (FFT) for a serial (unparallelised case), and stores the execution
# times as a function of problem size (N)

import numpy as np
import sys
import os

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from timing import time_function

# Define the problem sizes as powers of 2 from 2^0 to 2^8
problem_sizes = [2**i for i in range(13)] 
num_runs = 10  # Number of runs for averaging

# Run the serial 1D FFT for each problem size and measure time
for size in problem_sizes:
    data = np.random.random(size)  # Generate random 1D data for FFT

    # Measure execution time over multiple runs and calculate mean
    times = [time_function(np.fft.fft, data) for _ in range(num_runs)]
    mean_time = np.mean(times)

    # Print the execution time for the current problem size
    print(f"1D (Serial) FFT Execution Time for N={size}: {mean_time:.10f} seconds")
