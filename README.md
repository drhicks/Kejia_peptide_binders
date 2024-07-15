# Kejia_peptide_IDR_binders

A computational pipeline to target arbitrary unstructured sequence fragments (4-30 amino acids) of intrinsically disordered proteins and peptides, with de novo designed binding proteins.

## Prerequisites

- A Python environment with PyRosetta.
- Make sure silent_tools is in your PATH
- Download DL weights for AlphaFold, ProteinMPNN, and RF_Diffusion.
- Correct paths in MPNN scripts for your local installs.
- Update paths in `path_to/threading/make_jobs.py` to use your Python environment.
- Ensure `path_to/threading/make_jobs.py` has the correct path for `path_to/threading/thread_peptide_sequence_new.py`

## Step-by-Step Guide

### from your working directory for this binder project.

### 1. Threading

1. **Make a directory for threading**:
    ```sh
    mkdir 1_threading
    cd 1_threading
    ```

2. **Make all jobs**:
    make your target fasta file
    make your template list file
    ```sh
    python path_to/threading/make_jobs.py path_to/peptide.fasta path_to/templates.list | sort -R > all_jobs
    ```

3. **Split jobs into smaller sets**:
    ```sh
    split -l 3 all_jobs
    ```

4. **Add all job sets to SLURM array list**:
    ```sh
    for i in x* ; do echo "bash $i" ; done > jobs
    ```

5. **Submit array file jobs to SLURM**:
    ```sh
    path_to/threading/submit_jobs.sh
    ```

6. **After jobs finish, collect all PDB outputs into a silent file**:
    ```sh
    silentfrompdbs path_to_pdbs/*pdb > threading.silent
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

    or (with apptainer)
    ```sh
    path_to/job_creation/mpnn_design_job_create -prefix mpnn -script path_to/mpnn_git_repo/design_scripts/killer_mpnn_interface_design.py -p cpu -t 12:00:00 -mem 5 -cpus 1 -apptainer path_to/your_apptainer -structs_per_job 100 -silent path_to/threading.silent -args "--num_seq_per_target 5 --max_out 5 --sampling_temp 0.1"
    ./run_submit.sh
    ```

    In the paper, we have been routinely doing 2-rounds sequence design (i.e., MPNN-relax-MPNN-relax) for many targets in an ealier time. To do this, one can simply run the above script twice with the flag --relax on. Set --num_seq_per_target 1 in the first run and --num_seq_per_target 5 in the second.
    

4. **Concatenate all the silent files together**:
    ```sh
    cat mpnn_runs/*/*silent > mpnn_out.silent
    ```

### 3. AlphaFold Filtering and Refinement (AF2-initial-multimer)

1. **Back in your original working directory**:
    ```sh
    mkdir 3_af2_im
    cd 3_af2_im
    ```

2. **Make array jobs**:
    ```sh
    path_to/job_creation/interfaceaf2create -prefix af2 -script path_to/colabfold_initial_guess/AlphaFold2_initial_guess_multimer.py -silent ../2_mpnn/mpnn_out.silent -gres "gpu:1" -apptainer /home/drhicks1/scripts/Kejia_peptide_binders/colabfold_initial_guess/make_apptainer/colab_fold_ig.sif -structs_per_job 300 -p gpu-bf -t 06:00:00
    ./run_submit.sh
    ```

3. **Concatenate all the silent files together**:
    ```sh
    cat af2_runs/*/*silent > af2_out.silent
    ```

4. **Create a scorefile**:
    ```sh
    silent_tools/silentscorefile af2_out.silent
    ```

5. **Filter with sequence clustering and picking the top AlphaFold output/s per cluster after averaging 5 models**:
    ```sh
    python path_to/af2_filtering/average_af2_model_scores.py af2_out.sc > af2_out_averaged.sc
    python path_to/af2_filtering/dynamic_filtering_by_group.py af2_out_averaged.sc af2_out.silent > cluster.log
    column_number=$(head -1 cluster.log | tr '\t' '\n' | grep -n 'description' | cut -d: -f1); awk -v col=$column_number 'NR > 1 {print $col}' cluster.log | grep -oE '[a-zA-Z0-9_]+_af2mv3_[0-9]+' > tags
    ```

### 4. MPNN/AF2 cycle
1. **Repeat MPNN step on filtered silent file**.

2. **Repeat AlphaFold IM and filtering**.

### 5. Sequence Only AlphaFold Filtering and Refinement (potentially optional)
1. **Make fasta file**.
    ```sh
    silentsequence path_to/af2_out_filtered.silent | awk '{print ">"$3"\n"$1":"$2}' > colabfold_input.fasta
    ```
    
2. **Run colabfold**.
    ```sh
    /home/drhicks1/scripts/Kejia_peptide_binders/colabfold_initial_guess/make_apptainer/colab_fold_ig.sif AlphaFold2_jupyter-batch_hack_new_v2.py --fasta colabfold_input.fasta --num_recycles 10
    ```

3. **Concatenate all the silent files together**:
    ```sh
    cat af2_runs/*/*silent > af2_out.silent
    ```

4. **Create a scorefile**:
    ```sh
    silent_tools/silentscorefile af2_out.silent
    ```

5. **Filter with sequence clustering and picking the top AlphaFold output/s per cluster after averaging 5 models**:
    ```sh
    python path_to/af2_filtering/average_af2_model_scores.py af2_out.sc > af2_out_averaged.sc
    python path_to/af2_filtering/dynamic_filtering_by_group.py af2_out_averaged.sc af2_out.silent --not_initial_guess
    ```

### Additional Steps

- **Other filtering steps as desired such as af2_filtering/rosetta_min_ddg.py and visual inspection to order**.
- **Incorporate motif diffusion or partial diffusion as needed**

## Notes

- You may choose to run MPNN with Rosetta relax, but this is slow and questionably useful. If you do, add the flag `--relax`.
- If you run relax or minimization methods that can perturb the rigid body and/or binder/target backbones, you could run a second MPNN on the output of the first to potentially design a better sequence. However, the current preference is to go straight to AlphaFold filtering/refinement.
- Diffusion can be run at various steps such as:
1. After 1 or 2 rounds of mpnn/af2 , if not enough designs (< 70) passing the final filtering criteria, in which case you will want to repeat the two cycles of mpnn/af2 on the output from diffusion.
2. On the final designs before ordering, if enough designs (>= 70) passing the final filtering criteria, but one may want to order on chips and/or include arbitrary refined designs in initial test, in which case you can repeat either the one cycle or two cycles of mpnn/af2 on the output from diffusion. Depending on available computation resources and chip quota.
3. On the initial hits after experimental screeining and characterization, in which case you will want to repeat the two cycles of mpnn/af2 on the output from diffusion.
- In general the pipeline works most times without the use of diffusion, however, intellegent use of diffusion can increase in silico success rates for difficult targets and potentially improve the affinity and specificty of characterized binders.
- There are many knobs that can be tuned and variations of the pipeline that can be run depending on the ease or difficulty of individual targets.

