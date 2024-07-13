#!/bin/bash

# This script will submit an array job to SLURM based on the number of jobs listed in the 'jobs' file.

#SBATCH -p cpu
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5g
#SBATCH -o run.log

# Check if the 'jobs' file exists
if [ ! -f jobs ]; then
  echo "Error: 'jobs' file not found."
  exit 1
fi

# Get the number of jobs
NUM_JOBS=$(wc -l < jobs)

# Submit the array job
sbatch --array=1-$NUM_JOBS --wrap="sed -n \${SLURM_ARRAY_TASK_ID}p jobs | bash"

