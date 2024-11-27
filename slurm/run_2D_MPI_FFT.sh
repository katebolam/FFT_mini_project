#!/bin/bash
# ======================
# 2D MPI FFT SLURM script
# ======================
#SBATCH --job-name=2D_MPI_FFT
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=<num_nodes> # Replace <num_nodes> with number of nodes for processes to be distributed over
#SBATCH --ntasks_per_node=<num_processes_per_node>  # Replace <num_processes_per_node> with desired process count per node
#SBATCH --cpus-per-task= 1
#SBATCH --time=0:10:00
#SBATCH --mem=1G

# Load the required Python module
module add languages/python/3.12.3

# Navigate to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run the 1D MPI FFT
mpiexec -np $SLURM_NTASKS python src/MPI/2D_MPI_FFT.py