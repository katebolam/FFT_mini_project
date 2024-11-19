import numpy as np
import matplotlib.pyplot as plt

# Define the number of processes as powers of 2
processes = 2 ** np.arange(13)  # 2^1 to 2^12

# Define execution times for 1D and 2D FFT
execution_time_1d_linear = np.array([
    0.0000056900,  # N=1
    0.0000070200,  # N=2
    0.0000046900,  # N=4
    0.0000050100,  # N=8
    0.0000047600,  # N=16
    0.0000064600,  # N=32
    0.0000046900,  # N=64
    0.0000055800,  # N=128
    0.0000116200,  # N=256
    0.0000170500,  # N=512
    0.0000206800,  # N=1024
    0.0000597900,  # N=2048
    0.0000740900   # N=4096
])

execution_time_2d_linear = np.array([
    0.0000212000,  # 1x1
    0.0000570200,  # 2x2
    0.0000329800,  # 4x4
    0.0000198000,  # 8x8
    0.0000242600,  # 16x16
    0.0000435600,  # 32x32
    0.0000858500,  # 64x64
    0.0005473000,  # 128x128
    0.0019678000,  # 256x256
    0.0080807000,  # 512x512
    0.0380911400,  # 1024x1024
    0.2085887700,  # 2048x2048
    0.8956712000   # 4096x4096
])

# Calculate log10 values of execution times
execution_time_1d_log = np.log10(execution_time_1d_linear)
execution_time_2d_log = np.log10(execution_time_2d_linear)

# --- Linear Scale Plot ---
plt.figure(figsize=(8, 5))
# Solid lines for linear scale
plt.plot(processes, execution_time_1d_linear, 'r-', label="1D serial FFT")
plt.plot(processes, execution_time_2d_linear, 'b-', label="2D serial FFT")


plt.xlabel("N(xN)")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2) 
plt.legend()


# Show linear plot
plt.show()

# --- Logarithmic Scale Plot ---
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_time_1d_log, 'r--', label="1D serial FFT")
plt.plot(processes, execution_time_2d_log, 'b--', label="2D serial FFT")

plt.xlabel("N(xN)")
plt.ylabel("log(Execution Time) (s)")
plt.legend()
plt.xscale('log', base=2)  # Log scale for the x-axis with base 2

# Show log scale plot
plt.show()
