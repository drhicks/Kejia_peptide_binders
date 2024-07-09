#!/bin/bash 
#SBATCH -p cpu
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5g
#SBATCH -o run.log
sed -n ${SLURM_ARRAY_TASK_ID}p jobs | bash
