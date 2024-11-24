import numpy as np
import matplotlib.pyplot as plt

# Define the number of processes as powers of 2
processes = 2 ** np.arange(12)  # 2^0 to 2^11

# Define execution times for 1D and 2D FFT
execution_time_1d_linear_i5 = np.array([
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
    0.0000604000
])

execution_time_1d_linear_bc = np.array(
    [6.26800e-06, 
     4.18220e-06,
     3.46020e-06, 
     2.87330e-06, 
     2.91540e-06,
     3.08960e-06, 
     6.55260e-06, 
     4.00880e-06, 
     5.03320e-06, 
     9.41830e-06,
     1.39521e-05, 
     3.02183e-05
])

execution_time_2d_linear_i5 = np.array([
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

execution_time_2d_linear_bc = np.array([
    2.28763e-05, 
    1.89539e-05, 
    1.98456e-05, 
    1.98822e-05, 
    1.97466e-05, 
    4.18754e-05, 
    7.38969e-05, 
    2.620971e-04, 
    1.0741342e-03, 
    4.9309270e-03, 
    2.78509304e-02, 
    1.726563882e-01
])


# --- Linear 1D Scale Plot ---
plt.figure(figsize=(8, 5))
# Solid lines for linear scale
plt.plot(processes, execution_time_1d_linear_i5, 'r-', label="1D serial i5")
plt.plot(processes, execution_time_1d_linear_bc, 'b-', label="1D serial BC4")
plt.xlabel("N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2) 
plt.legend()
plt.show()

# --- Log 1D Scale Plot ---
plt.figure(figsize=(8, 5))
# Solid lines for linear scale
plt.plot(processes, execution_time_1d_linear_i5, 'r--', label="1D serial i5")
plt.plot(processes, execution_time_1d_linear_bc, 'b--', label="1D serial BC4")
plt.xlabel("N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2) 
plt.yscale('log')
plt.minorticks_off()
plt.legend()
plt.show()


# --- Linear 2D Scale Plot ---
plt.figure(figsize=(8, 5))
# Solid lines for linear scale
plt.plot(processes, execution_time_2d_linear_i5, 'r-', label="2D serial i5")
plt.plot(processes, execution_time_2d_linear_bc, 'b-', label="2D serial BC4")
plt.xlabel("N x N")
plt.ylabel("Execution Time (s)")
plt.xscale('log', base=2) 
plt.legend()
plt.show()

# --- Log 2D Scale Plot ---
plt.figure(figsize=(8, 5))
plt.plot(processes, execution_time_2d_linear_i5, 'r--', label="2D serial i5")
plt.plot(processes, execution_time_2d_linear_bc, 'b--', label="2D serial BC4")
plt.xlabel("N x N")
plt.ylabel("Execution Time (s)")
plt.yscale('log')
plt.minorticks_off()
plt.legend()
plt.xscale('log', base=2)  
plt.show()
