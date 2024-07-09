#!/bin/bash
#SBATCH --mem=16g
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --output=run_packer.out

source activate /home/justas/.conda/envs/mlfold-test/

#Folder with PDBs to be packed
pdb_folder="/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/latest_sc/pdb_examples"

#Path to output parsed PDBs
parsed_output="/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/latest_sc/pdb_examples/parsed.jsonl"

#Path to output packed PDBs
packed_output="/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/latest_sc/outputs_max"


python ./parse_multiple_chains.py --input_path=$pdb_folder --output_path=$parsed_output
python ./script_to_pack.py $parsed_output $packed_output
