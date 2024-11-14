# Parallelizing FFTs

Parallel Fast Fourier Transforms

## Overview

This repository contains an implementation of the radix-2 Cooley-Tukey Fast Fourier Transform (FFT) algorithm, parallelized to run on both shared memory (multicore) and distributed memory (multisocket) architectures. The project explores different parallelization techniques, developed and tested in the context of both four-core PC and high-performance computing environments.

The project includes four main strands, each supporting both 1D and 2D FFTs:

1. **Shared Memory Systems with OpenMP**
   - Suitable for single or dual-socket CPU systems.
   - Utilizes OpenMP for parallelization on multicore CPUs.

2. **Distributed Memory Systems with MPI**
   - Suitable for multisocket or multinode systems.
   - Uses MPI to distribute data across multiple nodes or cores.

3. **Multiprocessing for Shared Memory Systems**
   - Uses Python’s multiprocessing module to achieve parallelism on multicore systems without OpenMP.
   - Supports both 1D and 2D FFTs.

4. **Serial (Baseline) Implementation**
   - Provides a serial version of the FFT for benchmarking and comparison.

## Structure

This repository is organized into the following directories:

- **`MPI/`**: Contains MPI-based implementations for both 1D and 2D FFTs, targeting distributed memory systems.
  - `1D_MPI_FFT.py`: MPI parallelized 1D FFT.
  - `2D_MPI_FFT.py`: MPI parallelized 2D FFT.

- **`OpenMP/`**: Contains Cython-based implementations with OpenMP for shared memory parallelization.
  - `OpenMP_1D_FFT.pyx`: OpenMP parallelized 1D FFT.
  - `OpenMP_2D_FFT.pyx`: OpenMP parallelized 2D FFT.
  - `setup_1D.py`: Build script for compiling 1D OpenMP FFT.
  - `setup_2D.py`: Build script for compiling 2D OpenMP FFT.
  - `test_1D_OpenMP_FFT.py`: Measures and plots execution time for the 1D OpenMP FFT across various problem sizes.
  - `test_2D_OpenMP_FFT.py`: Measures and plots execution time for the 2D OpenMP FFT across various problem sizes.


- **`multiprocessing/`**: Contains FFT implementations using Python’s multiprocessing module for parallelism on shared memory systems.
  - `1D_multiprocessing_FFT.py`: Multiprocessing-based 1D FFT.
  - `2D_multiprocessing_FFT.py`: Multiprocessing-based 2D FFT.

- **`serial/`**: Baseline serial implementations of the FFT algorithm.
  - `1D_serial_FFT.py`: Serial 1D FFT.
  - `2D_serial_FFT.py`: Serial 2D FFT.

- **`utils/`**: Helper modules for timing and plotting, used for benchmarking and visualizing results.
  - `timing.py`: Functions for measuring execution time.
  - `plotting.py`: Functions for plotting execution time vs. problem size.

## Executables and Usage

Each parallelization technique supports both 1D and 2D FFTs, with four main executables for various configurations:

1. **(Shared Memory) OpenMP 1D FFT** (`OpenMP_1D_FFT`)
   - Built using `setup_1D.py` for Cython and OpenMP.

2. **(Shared Memory) OpenMP 2D FFT** (`OpenMP_2D_FFT`)
   - Built using `setup_2D.py` for Cython and OpenMP.

3. **(Distributed Memory) 1D FFT** (`1D_MPI_FFT`)
   - Uses MPI. Execute with `mpiexec` to distribute the computation.

4. **(Distributed Memory) 2D FFT** (`2D_MPI_FFT`)
   - Uses MPI. Execute with `mpiexec` for distributed computation.

Example commands to build and execute:
- To build OpenMP-based executables:
  ```bash
  python setup_1D.py build_ext --inplace
  python setup_2D.py build_ext --inplace
- To run the 2D OpenMP FFT:
  ```bash
  python OpenMP/test_2D_OpenMP_FFT.py
- To run the 1D MPI FFT:
  ```bash
  mpiexec -np <num_processes> python MPI/1D_MPI_FFT.py

## Benchmarking
The purpose of this program is to benchmark parallelization strategies of the Cooley-Tukey FFT algorithm. Each parallel implementation measures and records 
execution time for a range of problem sizes (powers of 2), which can be plotted for performance analysis.
