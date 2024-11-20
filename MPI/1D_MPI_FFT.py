from mpi4py import MPI
import numpy as np
from numpy.fft import fft
import sys
import os

# Add utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from timing import time_function
from plotting import plot_execution_times

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the problem sizes as powers of 2 from 2^0 to 2^12
problem_sizes = [2**i for i in range(13)]
execution_times = []  # To store mean execution times for each problem size
valid_sizes = []  # To store valid problem sizes for plotting

# Number of runs for averaging
num_runs = 1

# Scatter and collect times for each problem size
for N in problem_sizes:
    data = None
    if rank == 0:
        data = np.random.random(N)  # Generate random 1D data for FFT

    # Check if N is smaller than the number of processes or not divisible
    chunk_size = N // size
    if chunk_size == 0:
        print(f"Warning: N={N} is too small for {size} processes. Skipping...")
        continue  # Skip this N if it's too small

    local_data = np.zeros(chunk_size, dtype=np.float64)
    comm.Scatter(data, local_data, root=0)

    # Perform the FFT on the local chunk of data
    local_fft = fft(local_data)

    # Gather the results from all processes
    global_fft = None
    if rank == 0:
        global_fft = np.zeros(N, dtype=np.complex128)
    comm.Gather(local_fft, global_fft, root=0)

    # Measure execution time over multiple runs and calculate mean
    if rank == 0:
        times = [time_function(fft, data) for _ in range(num_runs)]
        mean_time = np.mean(times)
        execution_times.append(mean_time)
        valid_sizes.append(N)  # Add only valid N values
        print(f"1D (MPI) FFT Execution Time for N={N}: {mean_time:.10f} seconds")

# Plot the execution times only if execution_times has data
if rank == 0 and execution_times:
    plot_execution_times(
        valid_sizes,  # Use valid sizes that were processed
        [execution_times],  
        labels=["MPI 1D FFT"], 
        title="1D FFT MPI Execution Time vs. Problem Size",
        xlabel="Problem Size (2^N)",
        ylabel="Execution Time (seconds)"
    )
else:
    if rank == 0:
        print("No data to plot.")
