#!/bin/bash
#SBATCH --job-name=speech_job         # Job name
#SBATCH --partition=fat               # Choose the appropriate partition fat

#SBATCH --ntasks=1                      # Run a single task
            
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gres=gpu:2                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=19:00:00                 # Time limit hrs:min:sec
#SBATCH --output=speech_assignment2.log        # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie218@iitj.ac.in  # Your email address for notifications

export OMP_NUM_THREADS=1
# Activate Conda environment
source ~/.bashrc                                 # Ensure Conda is initialized
conda activate speech_env

# Load required Python module
module load openmpi4                  # Adjust version as necessary

# Print useful job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

# Set CUDA memory allocation policy for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Run the Python script
#python creat_transcript.py
#python create_syllebus.py
python pipeline.py

# Print completion message
echo "Job completed successfully at: $(date)"

conda deactivate