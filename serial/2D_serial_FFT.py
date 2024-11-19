# this script implements a 2D fast-fourier transform (FFT) for a serial (unparallelised case), and stores the execution
# times as a function of problem size (NxN)

import numpy as np
import sys
import os

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from timing import time_function

# Define the problem sizes as powers of 2 from 2^0 to 2^8
problem_sizes = [2**i for i in range(13)] 
num_runs = 10  # Number of runs for averaging

# Run the serial 2D FFT for each problem size and measure time
for size in problem_sizes:
    data = np.random.random((size, size))  # Generate random 2D data for FFT

    # Measure execution time over multiple runs and calculate mean
    times = [time_function(np.fft.fft2, data) for _ in range(num_runs)]
    mean_time = np.mean(times)

    # Print the execution time for the current problem size
    print(f"2D (Serial) FFT Execution Time for {size}x{size}: {mean_time:.10f} seconds")
