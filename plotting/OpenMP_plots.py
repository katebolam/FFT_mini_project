import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df_1D = pd.read_csv("C:\\Users\\Kate\\OneDrive\\Documents\\Uni\\Y4\\Advanced Computational Physics\\FFT_mini_project\\plotting\\1D_OpenMP_FFT_times.csv")
df_2D = pd.read_csv("C:\\Users\\Kate\\OneDrive\\Documents\\Uni\\Y4\\Advanced Computational Physics\\FFT_mini_project\\plotting\\2D_OpenMP_FFT_times.csv")

# Plot for 1D FFT

# linear
plt.figure(figsize=(8, 5))
problem_sizes_1D = df_1D['Problem Size']
threads_columns_1D = df_1D.columns[1:]
for column in threads_columns_1D:
    linestyle = '--' if column == 'p=1' else '-'  # Make 'p=1' dashed
    plt.plot(problem_sizes_1D, df_1D[column], label=column, linestyle=linestyle)
plt.xlabel("N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2)
plt.legend()
plt.show()

# log
plt.figure(figsize=(8, 5))
problem_sizes_1D = df_1D['Problem Size']
threads_columns_1D = df_1D.columns[1:]
for column in threads_columns_1D:
    linestyle = '--' if column == 'p=1' else '-'  # Make 'p=1' dashed
    plt.plot(problem_sizes_1D, df_1D[column], label=column, linestyle=linestyle)
plt.xlabel("N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2)
plt.yscale('log')
plt.minorticks_off()
plt.legend()
plt.show()

# Plot for 2D FFT

# linear
plt.figure(figsize=(8, 5))
problem_sizes_2D = df_2D['Problem Size']
threads_columns_2D = df_2D.columns[1:]
for column in threads_columns_2D:
    linestyle = '--' if column == 'p=1' else '-'  # Make 'p=1' dashed
    plt.plot(problem_sizes_2D, df_2D[column], label=column, linestyle=linestyle)
plt.xlabel("N x N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2)
plt.legend()
plt.show()

# log
plt.figure(figsize=(8, 5))
problem_sizes_2D = df_2D['Problem Size']
threads_columns_2D = df_2D.columns[1:]
for column in threads_columns_2D:
    linestyle = '--' if column == 'p=1' else '-'  # Make 'p=1' dashed
    plt.plot(problem_sizes_2D, df_2D[column], label=column, linestyle=linestyle)
plt.xlabel("N x N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2)
plt.yscale('log')
plt.minorticks_off()
plt.legend()
plt.show()
