# Kejia_peptide_binders

# A computation pipeline to target arbitrary unstructured sequence fragments (8-30 amino acids) of intrinsically disordered proteins and peptides, with de novo designed proteins.

# you will need a python env with pyrosetta

# you will need to download DL weights for alphafold, proteinmpnn, and rf_diffusion

# correct paths in mpnn scripts for you local installs

# update paths in path_to/threading/make_jobs.py to use your python env
# make sure path_to/threading/make_jobs.py has the correct path path_to/threading/thread_peptide_sequence_v2_and_pert.py
# from your working directory for this binder project

# make a dir for threading
mkdir 1_threading ; cd 1_threading
# make all jobs
python path_to/threading/make_jobs.py path_to/peptide.fasta path_to/scaffolds.list | sort -R > all_jobs
# split jobs to smaller sets
split -l 10 all_jobs
# add all job sets to slurm array list
for i in x* ; do echo "bash $i" ; done > jobs

# submit array file jobs to slurm
bash path_to/threading/submit_runs.sh path_to/threading/sbatch_array.sh

# after jobs finish, collect all pdb outputs into a silent file
silent_tools/silentfrompdbs path_to_pdbs_*pdb > threading.silent

# back in your original working dir
mkdir 2_mpnn ; cd 2_mpnn
# you need a python with env that can run mpnn and has pyrosetta
# run the intial mpnn (without relax; preferred)

# create array jobs
path_to/job_creation/dev_mpnn_design_job_create -prefix mpnn -script path_to/mpnn_git_repo/design_scripts/killer_mpnn_interface_design.py -p cpu -t 12:00:00 -mem 5 -cpus 1 -conda path_to/env/mpnn_pyro -structs_per_job 100 -silent path_to/threading.silent -args "--num_seq_per_target 5 --max_out 5 --sampling_temp 0.1"

./run_submit.sh

#cat all the silent files together
cat mpnn_runs/*/*silent > mpnn_out.silent

# you could have also run mpnn with rosetta relax, but this is slow and questionably useful... just add the flag --relax
# if you run relax or minimization or some method that can perturb the rigid body and/or binder/target backbones, you could run a second mpnn on the output of the first to in theory design a better sequence (assuming the relax/min improved the binder in someway). This used to be the favored way to design binders, now we prefer to go straight to alphafold filtering/refinement

# Alphafold filtering and refinement
# you will need to make a python env or apptainer to run our colabfold initial guess with pyrosetta ie colabfold_initial_guess/make_apptainer/colab_fold_ig.spec

# back in your original working dir
mkdir 3_af2 ; cd 3_af2

# make array jobs
path_to/job_creation/interfaceaf2create -prefix af2_ig -script path_to/colabfold_initial_guess/AlphaFold2_initial_guess_multimer.py -silent ../2_mpnn/mpnn_out.silent -gres "gpu:1" -apptainer /software/containers/users/drhicks1/colabfold_ig/colab_fold_ig.sif -structs_per_job 300 -p gpu-bf -t 06:00:00

./run_submit.sh

# cat all the silent files together
cat af2_ig_runs/*/*silent > af2_ig_out.silent
# create a scorefile
silent_tools/silentscorefile af2_ig_out.silent

# filter with sequence clustering and picking the top af2 output/s per cluster after averaging 5 models
python path_to/af2_filtering/average_af2_model_scores.py af2_ig_out.sc > af2_ig_out_averaged.sc
python path_to/af2_filtering/dynamic_filtering_by_group.py af2_ig_out_averaged.sc af2_ig_out.silent

# some prosessing step on the filtering out to get filtered silent file...

# repeat mpnn step on filtered silent file

# repeat af2_ig and filtering

# sequence only colabfold

# some filtering and final visual inspection to order



