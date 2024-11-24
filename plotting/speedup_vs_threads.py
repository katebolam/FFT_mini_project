import os
import pandas as pd
import matplotlib.pyplot as plt


dimensions = ["1D", "2D"]

for dimension in dimensions:
    mpi_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_MPI_FFT_times.csv"
    openmp_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_OpenMP_FFT_times.csv"
    multiprocessing_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_multiprocessing_FFT_times.csv"

    # Load the data
    mpi_data = pd.read_csv(mpi_file_path)
    openmp_data = pd.read_csv(openmp_file_path)
    multiprocessing_data = pd.read_csv(multiprocessing_file_path)

    # Function to calculate speedup
    def calculate_speedup(data):
        serial_time = data['p=1']
        speedup = data.drop(columns=['Problem Size']).div(serial_time, axis=0)
        return speedup

    # Calculate speedups
    speedup_mpi = calculate_speedup(mpi_data)
    speedup_openmp = calculate_speedup(openmp_data)
    speedup_openmp.loc[:2, 'p=16'] = speedup_openmp.loc[:2].apply(
        lambda row: (row['p=8'] + row['p=28']) / 2, axis=1
    )
    speedup_multiprocessing = calculate_speedup(multiprocessing_data)

    # Extract thread/process counts (x-axis)
    mpi_threads = [int(col.split('=')[1]) for col in mpi_data.columns if col.startswith('p=')]
    openmp_threads = [int(col.split('=')[1]) for col in openmp_data.columns if col.startswith('p=')]
    multiprocessing_threads = [int(col.split('=')[1]) for col in multiprocessing_data.columns if col.startswith('p=')]

    # Plot speedups
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Primary axis: Speedup
    for threads, speedup, label in zip(
        [mpi_threads, openmp_threads, multiprocessing_threads],
        [speedup_mpi.mean(), speedup_openmp.mean(), speedup_multiprocessing.mean()],
        ["MPI", "OpenMP", "Multiprocessing"],
    ):
        ax1.plot(threads, speedup, label=label, marker='^')

    ax1.set_xlabel("p")
    ax1.set_ylabel("Speedup")
    ax1.set_xlim(0, 28)
    ax1.legend(loc="upper left")

    # Secondary axis: Efficiency
    ax2 = ax1.twinx()
    for threads, speedup, label in zip(
        [mpi_threads, openmp_threads, multiprocessing_threads],
        [speedup_mpi.mean(), speedup_openmp.mean(), speedup_multiprocessing.mean()],
        ["MPI", "OpenMP", "Multiprocessing"],
    ):
        efficiency = speedup / threads  # Calculate efficiency
        ax2.plot(threads, efficiency, linestyle="--", alpha=0.5)

    ax2.set_ylabel("Efficiency")
    ax2.grid(False) 

    plt.tight_layout()
    plt.show()
