import matplotlib.pyplot as plt

def plot_execution_times(problem_sizes, execution_times, labels, title, xlabel, ylabel):
    """
    Plots execution times for different methods or configurations with custom styling.
    
    Parameters:
    - problem_sizes: List of problem sizes (x-axis).
    - execution_times: List of lists containing execution times for each configuration (y-axis).
    - labels: List of labels for each configuration.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    for times, label in zip(execution_times, labels):
        plt.plot(
            problem_sizes, times, marker='s', linestyle='-', linewidth=0.5, color = 'black',
            markersize=2, label=label
        )  
    plt.xscale('log')  # Logarithmic scale for problem sizes
    
    # Generate labels as powers of 2
    indices_of_2 = [f"{i}" for i in range(len(problem_sizes))]
    plt.xticks(problem_sizes, labels=indices_of_2)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.minorticks_off() 
    plt.tight_layout()
    plt.show()
    
def plot_speedup(problem_sizes, serial_times, parallel_times, labels):
    """
    Plots speedup for parallel implementations compared to serial times.
    
    Parameters:
    - problem_sizes: List of problem sizes (x-axis).
    - serial_times: List of serial execution times (for reference).
    - parallel_times: List of lists containing parallel execution times.
    - labels: List of labels for each parallel configuration.
    """
    for times, label in zip(parallel_times, labels):
        speedup = [st / pt for st, pt in zip(serial_times, times)]
        plt.plot(problem_sizes, speedup, marker='o', label=f'Speedup ({label})')
    
    plt.title("Speedup of Parallel FFT Implementations")
    plt.xlabel("Problem Size")
    plt.ylabel("Speedup (Serial Time / Parallel Time)")
    plt.legend()
    plt.grid(True)
    plt.show()
