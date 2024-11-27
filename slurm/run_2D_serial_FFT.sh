#!/bin/bash
# ======================
# 2D Serial FFT SLURM script
# ======================
#SBATCH --job-name=2D_serial_FFT
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --mem=500M

# Load the required Python module
module add languages/python/3.12.3

# Navigate to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run the 2D Serial FFT
python src/serial/2D_serial_FFT.py
