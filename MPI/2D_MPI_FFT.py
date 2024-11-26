from mpi4py import MPI
import numpy as np
from numpy.fft import fft2
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

# Define the problem sizes as powers of 2 from 2^2 to 2^12
problem_sizes = [2**i for i in range(2, 13)]  # Start from 2 to avoid issues with size 1
execution_times = []  # To store mean execution times for each problem size
valid_sizes = []  # To store valid problem sizes for plotting

# Number of runs for averaging
num_runs = 1

# Scatter and collect times for each problem size
for N in problem_sizes:
    # Ensure the data size is divisible by the number of processes
    rows_per_process = N // size
    if rows_per_process == 0:
        print(f"Warning: N={N} is too small for {size} processes. Skipping...")
        continue

    data = None
    if rank == 0:
        data = np.random.random((N, N))  # Generate 2D random data for FFT
    
    # Scatter the data to all processes
    chunk_size = N * rows_per_process  # Total data size for each process
    local_data = np.zeros(chunk_size, dtype=np.float64)
    comm.Scatter(data.flatten() if rank == 0 else None, local_data, root=0)

    # Reshape the local data to a 2D matrix
    local_data_reshaped = local_data.reshape((rows_per_process, N))

    # Perform the FFT on the local chunk of data
    local_fft = fft2(local_data_reshaped)

    # Gather the results from all processes
    global_fft = None
    if rank == 0:
        global_fft = np.zeros((N, N), dtype=np.complex128)
    comm.Gather(local_fft, global_fft if rank == 0 else None, root=0)

    # Measure execution time over multiple runs and calculate mean
    if rank == 0:
        times = [time_function(fft2, data) for _ in range(num_runs)]
        mean_time = np.mean(times)
        execution_times.append(mean_time)
        valid_sizes.append(N)  # Add only valid N values
        print(f"2D (MPI) FFT Execution Time for N={N}: {mean_time:.10f} seconds")
