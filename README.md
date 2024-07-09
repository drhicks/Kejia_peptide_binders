# Kejia_peptide_binders

A computation pipeline to target arbitrary unstructured sequence fragments (8-30 amino acids) of intrinsically disordered proteins and peptides, with de novo designed proteins.

## Prerequisites

- A Python environment with PyRosetta.
- Download DL weights for AlphaFold, ProteinMPNN, and RF_Diffusion.
- Correct paths in MPNN scripts for your local installs.
- Update paths in `path_to/threading/make_jobs.py` to use your Python environment.
- Ensure `path_to/threading/make_jobs.py` has the correct path `path_to/threading/thread_peptide_sequence_v2_and_pert.py` from your working directory for this binder project.

## Step-by-Step Guide

### 1. Threading

1. **Make a directory for threading**:
    ```sh
    mkdir 1_threading
    cd 1_threading
    ```

2. **Make all jobs**:
    ```sh
    python path_to/threading/make_jobs.py path_to/peptide.fasta path_to/scaffolds.list | sort -R > all_jobs
    ```

3. **Split jobs into smaller sets**:
    ```sh
    split -l 10 all_jobs
    ```

4. **Add all job sets to SLURM array list**:
    ```sh
    for i in x* ; do echo "bash $i" ; done > jobs
    ```

5. **Submit array file jobs to SLURM**:
    ```sh
    bash path_to/threading/submit_runs.sh path_to/threading/sbatch_array.sh
    ```

6. **After jobs finish, collect all PDB outputs into a silent file**:
    ```sh
    silent_tools/silentfrompdbs path_to_pdbs_*pdb > threading.silent
    ```

### 2. MPNN

1. **Back in your original working directory**:
    ```sh
    mkdir 2_mpnn
    cd 2_mpnn
    ```

2. **Run the initial MPNN (without relax; preferred)**:
    ```sh
    path_to/job_creation/dev_mpnn_design_job_create -prefix mpnn -script path_to/mpnn_git_repo/design_scripts/killer_mpnn_interface_design.py -p cpu -t 12:00:00 -mem 5 -cpus 1 -conda path_to/env/mpnn_pyro -structs_per_job 100 -silent path_to/threading.silent -args "--num_seq_per_target 5 --max_out 5 --sampling_temp 0.1"
    ./run_submit.sh
    ```

3. **Concatenate all the silent files together**:
    ```sh
    cat mpnn_runs/*/*silent > mpnn_out.silent
    ```

### 3. AlphaFold Filtering and Refinement

1. **Back in your original working directory**:
    ```sh
    mkdir 3_af2
    cd 3_af2
    ```

2. **Make array jobs**:
    ```sh
    path_to/job_creation/interfaceaf2create -prefix af2_ig -script path_to/colabfold_initial_guess/AlphaFold2_initial_guess_multimer.py -silent ../2_mpnn/mpnn_out.silent -gres "gpu:1" -apptainer /software/containers/users/drhicks1/colabfold_ig/colab_fold_ig.sif -structs_per_job 300 -p gpu-bf -t 06:00:00
    ./run_submit.sh
    ```

3. **Concatenate all the silent files together**:
    ```sh
    cat af2_ig_runs/*/*silent > af2_ig_out.silent
    ```

4. **Create a scorefile**:
    ```sh
    silent_tools/silentscorefile af2_ig_out.silent
    ```

5. **Filter with sequence clustering and picking the top AlphaFold output/s per cluster after averaging 5 models**:
    ```sh
    python path_to/af2_filtering/average_af2_model_scores.py af2_ig_out.sc > af2_ig_out_averaged.sc
    python path_to/af2_filtering/dynamic_filtering_by_group.py af2_ig_out_averaged.sc af2_ig_out.silent
    ```

### Additional Steps

- **Repeat MPNN step on filtered silent file**.
- **Repeat AlphaFold IG and filtering**.
- **Sequence-only ColabFold**.
- **Final filtering and visual inspection to order**.

## Notes

- You may choose to run MPNN with Rosetta relax, but this is slow and questionably useful. If you do, add the flag `--relax`.
- If you run relax or minimization methods that can perturb the rigid body and/or binder/target backbones, you could run a second MPNN on the output of the first to potentially design a better sequence. However, the current preference is to go straight to AlphaFold filtering/refinement.

This structured format should make your README more readable and user-friendly. Let me know if you need any further adjustments!

