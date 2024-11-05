import time

def time_function(func, *args, num_runs=1, **kwargs):
    """
    Times the execution of a given function over multiple runs and returns the average time.
    
    Parameters:
    - func: The function to time.
    - *args: Positional arguments for the function.
    - num_runs: Number of times to run the function to get an average timing.
    - **kwargs: Keyword arguments for the function.
    
    Returns:
    - avg_time: The average time taken to execute the function.
    """
    start_time = time.perf_counter()
    for _ in range(num_runs):
        func(*args, **kwargs)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time
