import os
import pandas as pd
import matplotlib.pyplot as plt

dimensions = ["1D", "2D"]

for dimension in dimensions:

    mpi_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_MPI_FFT_times.csv"
    openmp_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_OpenMP_FFT_times.csv"
    multiprocessing_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_multiprocessing_FFT_times.csv"
    JIT_file_path = rf"C:\Users\Kate\OneDrive\Documents\Uni\Y4\Advanced Computational Physics\FFT_mini_project\plotting\{dimension}_JIT_FFT_times.csv"

    # Load the data
    mpi_data = pd.read_csv(mpi_file_path)
    openmp_data = pd.read_csv(openmp_file_path)
    multiprocessing_data = pd.read_csv(multiprocessing_file_path)
    JIT_data = pd.read_csv(JIT_file_path)

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
    speedup_JIT = calculate_speedup(JIT_data)
    print(speedup_JIT)

    # Extract problem sizes (x-axis)
    problem_sizes_mpi = mpi_data["Problem Size"]
    problem_sizes_openmp = openmp_data["Problem Size"]
    problem_sizes_multiprocessing = multiprocessing_data["Problem Size"]
    problem_sizes_JIT = JIT_data["Problem Size"]

    # Plot Speedup vs Problem Size
    plt.figure(figsize=(8, 5))
    for problem_sizes, speedup, label in zip(
        [problem_sizes_mpi, problem_sizes_openmp, problem_sizes_multiprocessing, problem_sizes_JIT],
        [speedup_mpi["p=8"], speedup_openmp["p=8"], speedup_multiprocessing["p=8"], speedup_JIT],
        ["MPI", "OpenMP", "Multiprocessing", "JIT"],
    ):
        plt.plot(problem_sizes, speedup, label=label, marker='^')

    plt.xscale("log", base=2)
    if dimension == "1D":
        plt.xlabel("N")
    else: 
        plt.xlabel("N x N")
    plt.ylabel("Speedup")
    plt.legend()
    plt.xlim(2**0,2**13)
    plt.tight_layout()
    plt.show()
