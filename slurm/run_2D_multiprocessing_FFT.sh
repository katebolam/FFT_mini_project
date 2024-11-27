#!/bin/bash
# ======================
# 2D Multiprocessing FFT SLURM script
# ======================
#SBATCH --job-name=2D_multiprocessing_FFT
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<num_processes>  # Replace <num_processes> with desired thread count
#SBATCH --time=0:10:00
#SBATCH --mem=1G

# Load the required Python module
module add languages/python/3.12.3

# Navigate to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run the 1D Multiprocessing FFT
python src/multiprocessing/2D_multiprocessing_FFT.py --processes $SLURM_CPUS_PER_TASK
