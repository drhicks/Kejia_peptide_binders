#!/bin/bash
#SBATCH --mem=16g
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --output=submit.out

source activate /home/justas/.conda/envs/mlfold-test/

python ./sc_run.py \
       --batch_size 1 \
       --output_npz 0 \
       --output_folder_path "./test" \
       --pdb_path "./input_structures/5L33.pdb" \
       --score_only 1 
