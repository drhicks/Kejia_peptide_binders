# Kejia_peptide_binders

#you will need a python env with pyrosetta
#update paths in path_to/threading/make_jobs.py to use your python env
#make sure path_to/threading/make_jobs.py has the correct path path_to/threading/thread_peptide_sequence_v2_and_pert.py
#from your working directory to run threading jobs... ie 1_threading

#make all jobs
python path_to/threading/make_jobs.py path_to/peptide.fasta path_to/scaffolds.list | sort -R > all_jobs
#split jobs to smaller sets
split -l 10 all_jobs
#add all job sets to slurm array list
for i in x* ; do echo "bash $i" ; done > jobs

#submit array file jobs to slurm
bash path_to/threading/submit_runs.sh path_to/threading/sbatch_array.sh

