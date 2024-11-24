import numpy as np
import matplotlib.pyplot as plt

# Define the number of processes as powers of 2
processes = 2 ** np.arange(12)  # 2^0 to 2^11

# Define execution times for 1D and 2D FFT
execution_time_1d_linear_serial = np.array([
    0.0000172000,  
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


execution_time_1d_linear_JIT = np.array([
    0.0000009000,  
    0.0000056000, 
    0.0000089000,  
    0.0000108000,  
    0.0000111000,  
    0.0000139000,  
    0.0002638000,  
    0.0008266000,  
    0.0031386000,  
    0.0127169000,  
    0.0494824000,  
    0.1842949000,   
])

execution_time_2d_linear_serial = np.array([
    0.0000531000,  
    0.0000526000,  
    0.0000425000,  
    0.0000867000,  
    0.0001032000,  
    0.0001783000,  
    0.0004642000,  
    0.0019401000,  
    0.0078036000,  
    0.0369520000,  
    0.1847571000,  
    0.8195974000   
])
execution_time_2d_linear_JIT = np.array([
    0.0000386000,
    0.0000342000,
    0.0000312000,
    0.0000527000,
    0.0000920000,
    0.0014154000,
    0.0173930000,
    0.1498284000,
    1.0457054000,
    9.8357278000,
    83.3982042000,
    666.0349002000
])



# --- 1D Plot ---
plt.figure(figsize=(8, 5))
# Solid lines for linear scale
plt.plot(processes, execution_time_1d_linear_serial, 'r--', label="1D serial FFT (i5)")
plt.plot(processes, execution_time_1d_linear_JIT, '->', color= 'orange', label="1D JIT FFT (i5)")
plt.xlabel("N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2) 
plt.yscale('log')
plt.minorticks_off()
plt.legend()
plt.show()

# --- 2D Plot ---
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_time_2d_linear_serial, 'r--', label="2D serial FFT (i5)")
plt.plot(processes, execution_time_2d_linear_JIT, '->', color='orange', label="2D JIT FFT (i5)")
plt.xlabel("N x N")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.yscale('log')
plt.minorticks_off()
plt.xscale('log', base=2)  # Log scale for the x-axis with base 2
plt.show()
