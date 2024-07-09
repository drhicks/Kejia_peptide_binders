# Benchmarking pipeline

Utilities to run and analyze hyperparameter scans with design scripts
on a standard set of benchmark problems. See "benchmarks.json" for a list of
benchmarks and their specifications (i.e. input pdb and contig string).

See `benchmark_success_rate.ipynb` for a demonstration of calculating and
plotting the diversity-adjusted success rate, a standardized metric for
evaluating design runs in silico.

## Pipeline mode

This will generate designs, MPNN them, compute AF2 and PyRosetta metrics,
perform pairwise TM-align and TM-score-based clustering, and compile results
into a single CSV/Dataframe.

    ./pipeline.py --benchmarks rsv5-1 \
        --num_per_condition 10 --num_per_job 2 --out run1/run1 \
        --args "diffuser.T=20|50 diffuser.aa_decode_steps=5|10" \
               "diffuser.T=100|200 diffuser.aa_decode_steps=20|40"

`pipeline.py` occupies an active process while it submits slurm jobs and waits
for them to complete. You can run it interactively on the head or a compute
node, or submit it with modest resources, e.g.

    sbatch --mem 100m --wrap './pipeline.py --benchmarks rsv5-1 \
        --num_per_condition 10 --num_per_job 2 --out run1/run1 \
        --args "diffuser.T=20|50 diffuser.aa_decode_steps=5|10" \
               "diffuser.T=100|200 diffuser.aa_decode_steps=20|40"'

## Manual operation

Wait for slurm jobs to finish between steps

Step 1. Sweep hyperparameters:

    ./sweep_hyperparam.py --benchmarks rsv5-1 \
        --num_per_condition 10 --num_per_job 2 --out run1/run1 \
        --args "diffuser.T=20|50 diffuser.aa_decode_steps=5|10" \
               "diffuser.T=100|200 diffuser.aa_decode_steps=20|40"

Step 2. MPNN 

(wait for step 1 jobs to finish)

    ./mpnn_designs.py --chunk 20 run1/

(wait for jobs to finish)

    ./thread_mpnn.py run1/

Step 3. Score outputs:

    ./score_designs.py --chunk 20 run1/mpnn/

Step 4. Compile metrics:

(wait for step 3 jobs to finish)

    ./compile_metrics.py run1

Step 1 runs the design script on the "rsv5-1" benchmark with all
combinations of T in {20, 50} and aa_decode_steps in {5, 10}, plus all
combinations of T in {100, 200} and aa_decode_steps in {20, 40}.

Step 2 runs MPNN on all the designs from step 1. thread_mpnn.py generates dummy
PDBs with the diffusion structure but the MPNN sequence. This is used to
compute an RMSD to the AF2 prediction in the next step.

Step 3 runs AF2 monomer predictions and PyRosetta metrics on a folder of
designs, splitting them across multiple jobs with 20 designs each, and save
CSVs with metrics such as pLDDT and motif RMSDs. NOTE: In practice you want
--chunk 100 or more. This can be run on the non-mpnn designs as well (not shown
above).

Step 4 combines all completed AF2 metrics, including those from the MPNN
designs, if they exist, into a single table, adding design run metadata from
the trb file (each command-line flag becomes a column), and saves it to
combined_metrics.csv.


### Diversity-adjusted success rate

To obtain a final benchmark metric that represents success rate at varying
metric thresholds and structural diversities:

Pairwise TM-scores:

    ./pair_tmalign.py run1/

Wait for jobs to finish, then cluster on TM-scores:

    ./parse_tmalign.py run1/
    ./compile_metrics.py run1/

In Jupyter (`columns` should be whatever variables you want to group on to
calculate success rate):

    df = pd.read_csv('run1/compiled_metrics.csv')

    sys.path.append('/path/to/BFF/rf_diffusion/benchmark/util/')
    import analysis_util

    success = analysis_util.calc_success_rate(df, columns=['diffuser.T','diffuser.aa_decode_steps'])

See `benchmark_success_rate.ipynb` for an example of calculating and plotting success rates.


## Notes

General:

 - By default, sweep_hyperparams.py will submit an array job to slurm and
   print the slurm job ID.  
 - `--no_submit` will just print a list of jobs to stdout without submitting to
   slurm. Use this as a dry run to check your command.
 - Slurm logs are saved to working directory unless --no_logs is set.
 - Use the `-p` and `--gres` flags to optimize where you're running design &
   scoring jobs. Usually it's good to see which GPU nodes are free with `snodes
   -p gpu` and then send your jobs to those.  
 - The lists of jobs used for the slurm array is stored in the output folder.
   you can inspect it and manually resubmit it if needed (see
   `gpu_array_submit.sh` and `cpu_array_submit.sh`)

`sweep_hyperparams.py`

 - this puts a file jobs.list inside the output folder. you can manually submit
   this job later if something goes wrong.
 - this also copies any input pdbs for this job into a subfolder `input/` in
   the output folder, so scoring jobs can look for them there if the original
   path has been broken.
 - when resubmitting a partially complete job, it's useful to give the design
   script `inference.cautious=True` so it won't spend GPU time making designs
   that already exist.

`mpnn_designs.py` 

 - this puts a file jobs.mpnn.list inside the output folder. you can manually
   submit this job later if something goes wrong.
 - this also generates lists of design names to process in separate jobs. these
   are named `OUTPUT_FOLDER/mpnn/parse_multiple_chains.list.*`
 - this does preprocessing for mpnn inputs also as an array job. the mpnn job
   itself will be submitted with a dependency on the preprocessing jobs
 - if a job failed and you're resubmitting, you can use `--cautious` for this
   script to not process any designs for which a .fa already exists

`score_designs.py`

 - this generates files with lists of designs to score
 - this runs array jobs of various scoring scripts:
    - pyrosetta_metrics.py: gets radius of gyration, SS content, etc
    - af2_metrics.py: makes AF2 prediction and saves RMSDs and various quality metrics
