#!/bin/bash
# ======================
# 1D OpenMP FFT SLURM script
# ======================
#SBATCH --job-name=1D_OpenMP_FFT
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<num_threads>  # Replace <num_threads> with desired thread count
#SBATCH --time=0:10:00
#SBATCH --mem=500M

# Load the required Python module
module add languages/python/3.12.3

# Navigate to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run the 1D OpenMP FFT
python src/OpenMP/run_1D_OpenMP_FFT.py --threads $SLURM_CPUS_PER_TASK
