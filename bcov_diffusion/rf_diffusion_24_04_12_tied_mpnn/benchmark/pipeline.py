#!/usr/bin/env python
#
# Runs the benchmarking pipeline, given arguments for a hyperparameter sweep
#

import sys, os, re, subprocess, time, argparse
script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
    
def main():
    # parse --out argument for this script
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='out/out',help='Path prefix for output files')
    parser.add_argument('--start_step', type=str, default='sweep', choices=['sweep','mpnn','score'],
        help='Step of pipeline to start at')
    parser.add_argument('--inpaint', action='store_true', default=False, 
        help="Use sweep_hyperparam_inpaint.py, i.e. command-line arguments are in argparse format")
    parser.add_argument('--af2_unmpnned', action='store_true', default=False)
    parser.add_argument('--num_seq_per_target', default=8,type=int, help='How many mpnn sequences per design? Default = 8')
    args, unknown = parser.parse_known_args()

    outdir = os.path.dirname(args.out)
    job_id_tmalign=None

    if args.start_step == 'sweep':
        arg_str = ' '.join(['"'+x+'"' if (' ' in x or '|' in x or x=='') else x for x in sys.argv[1:]])
        if args.inpaint:
            script = f'{script_dir}sweep_hyperparam_inpaint.py'
        else:
            script = f'{script_dir}sweep_hyperparam.py'
        jobid_sweep = run_pipeline_step(f'{script} {arg_str}')

        print('Waiting for design jobs to finish...')
        wait_for_jobs(jobid_sweep)

    if args.start_step in ['sweep','mpnn']:
        jobid_mpnn = run_pipeline_step(f'{script_dir}mpnn_designs.py --num_seq_per_target {args.num_seq_per_target} --chunk 100 -p cpu --gres "" {outdir}')

        jobid_tmalign = run_pipeline_step(f'{script_dir}pair_tmalign.py {outdir}')

    if args.start_step in ['sweep','mpnn']:
        print('Waiting for MPNN jobs to finish...')
        wait_for_jobs(jobid_mpnn)

        print('Threading MPNN sequences onto design models...')
        run_pipeline_step(f'{script_dir}thread_mpnn.py {outdir}')
    if args.af2_unmpnned:
        jobid_score = run_pipeline_step(f'{script_dir}score_designs.py --chunk 100 {outdir}/')
    jobid_score_mpnn = run_pipeline_step(f'{script_dir}score_designs.py --chunk 100 {outdir}/mpnn')

    if job_id_tmalign:
        print('Waiting for TM-align jobs to finish...')
        wait_for_jobs(jobid_tmalign)

        print('Clustering by TM-score...')
        run_pipeline_step(f'{script_dir}parse_tmalign.py {outdir}')

    print('Waiting for scoring jobs to finish...')
    if args.af2_unmpnned:
        wait_for_jobs(jobid_score)
    wait_for_jobs(jobid_score_mpnn)

    print('Compiling metrics...')
    run_pipeline_step(f'{script_dir}compile_metrics.py {outdir}')

    print('Done.')
    
def run_pipeline_step(cmd):
    '''Runs a script in shell, prints its output, quits if there's an error,
    and returns list of slurm ids that appear in its output'''

    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out = proc.stdout.decode()
    print(out)

    if proc.returncode != 0: 
        sys.exit(proc.stderr.decode())

    jobids = re.findall(r'array job (\d+)', out)

    return jobids

def is_running(job_ids):
    '''Returns list of bools corresponding to whether each slurm ID in input
    list corresponds to a currently queued/running job.'''

    idstr = ','.join(map(str,job_ids))

    proc = subprocess.run(f'squeue -j {idstr}', shell=True, stdout=subprocess.PIPE)
    stdout = proc.stdout.decode()

    out = [False]*len(job_ids)
    for line in stdout.split('\n'):
        for i,id_ in enumerate(job_ids):
            if line.startswith(str(id_)):
                out[i] = True

    return out

def wait_for_jobs(job_ids, interval=60):
    '''Returns when all the SLURM jobs given in `job_ids` aren't running
    anymore.'''
    while True:
        if any(is_running(job_ids)):
            time.sleep(interval)
        else:
            break
    return 

if __name__ == "__main__":
    main()
