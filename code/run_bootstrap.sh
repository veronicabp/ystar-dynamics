#!/bin/bash
#SBATCH --job-name=main                    # Job name
#SBATCH --output=../log/output_%A_%a.out    # Output file
#SBATCH --error=../log/error_%A_%a.err      # Error file
#SBATCH --nodes=25                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # One task per node
#SBATCH --cpus-per-task=16                  # CPUs per task
#SBATCH --time=12:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=sched_mit_hill                # Partition name
#SBATCH --mem=60G                           # Memory per node

# Activate your Conda environment
source ~/.bashrc
conda activate ystar

# Run the Python script using mpirun
mpirun -np 25 python3 run_bootstrap.py