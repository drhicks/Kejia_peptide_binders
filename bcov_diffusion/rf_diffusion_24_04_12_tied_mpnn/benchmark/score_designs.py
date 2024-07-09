#!/usr/bin/env python
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import sys, os, argparse, itertools, json, glob
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'util'))
import slurm_tools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=-1,help='How many designs to score in each job')
    parser.add_argument('--tmp_pre',type=str,default='score.list', help='Name prefix of temporary files with lists of designs to score')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:rtx2080:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--pipeline', '-P', action='store_true', default=False, help='Pipeline mode: submit the next script to slurm with a dependency on jobs from this script.')

    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')

    filenames = sorted(glob.glob(args.datadir+'/*.pdb'))
    if len(filenames)==0: sys.exit('No pdbs to score. Exiting.')

    if args.chunk == -1:
        args.chunk = len(filenames)

    # AF2 predictions
    job_fn = args.datadir + '/jobs.score.af2.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for i in np.arange(0,len(filenames),args.chunk):
        tmp_fn = f'{args.datadir}/{args.tmp_pre}.{i}'
        with open(tmp_fn,'w') as outf:
            for j in np.arange(i,min(i+args.chunk, len(filenames))):
                print(filenames[j], file=outf)
        print(f'source activate /home/jue/.conda/envs/ampere; python {script_dir}/util/af2_metrics.py --use_ptm '\
              f'--outcsv {args.datadir}/af2_metrics.csv.{i} '\
              f'{tmp_fn}', file=job_list_file)

    # submit job
    if args.submit: 
        job_list_file.close()
        if args.J is not None:
            job_name = args.J 
        else:
            job_name = 'af2_'+os.path.basename(args.datadir.strip('/'))
        af2_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name)
        print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to AF2-predict {len(filenames)} designs')

    # pyrosetta metrics (rog, SS)
    job_fn = args.datadir + '/jobs.score.pyr.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for i in np.arange(0,len(filenames),args.chunk):
        tmp_fn = f'{args.datadir}/{args.tmp_pre}.pyr.{i}'
        with open(tmp_fn,'w') as outf:
            for j in np.arange(i,min(i+args.chunk, len(filenames))):
                print(filenames[j], file=outf)
        print(f'{script_dir}/util/pyrosetta_metrics.py '\
              f'--outcsv {args.datadir}/pyrosetta_metrics.csv.{i} '\
              f'{tmp_fn}', file=job_list_file)

    # submit job
    if args.submit: 
        job_list_file.close()
        if args.J is not None:
            job_name = args.J 
        else:
            job_name = 'pyr_'+os.path.basename(args.datadir.strip('/'))
        pyr_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=args.keep_logs, J=job_name)
        print(f'Submitted array job {pyr_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to get PyRosetta metrics for {len(filenames)} designs')


if __name__ == "__main__":
    main()
