import numpy as np
import matplotlib.pyplot as plt

# Define the number of processes as powers of 2
processes = 2 ** np.arange(12)  # 2^0 to 2^11

# Define execution times for 1D and 2D FFT
execution_time_1d_linear = np.array([
    0.0001522000,  
    0.0000174000,  
    0.0000148000, 
    0.0000120000,  
    0.0000164000,  
    0.0000153000,  
    0.0000113000, 
    0.0000115000,  
    0.0000127000,  
    0.0000164000,  
    0.0000374000,  
    0.0000604000, 
])


execution_time_2d_linear = np.array([
    0.0014131000,  
    0.0000526000, 
    0.0000425000,  
    0.0000567000,  
    0.0000832000,  
    0.0001483000,  
    0.0004642000, 
    0.0019401000,  
    0.0078036000,  
    0.0369520000,  
    0.1847571000,  
    0.8195974000  
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
