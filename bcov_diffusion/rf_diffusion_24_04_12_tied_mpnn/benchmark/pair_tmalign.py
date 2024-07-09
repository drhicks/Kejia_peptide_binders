#!/usr/bin/env python
#
# Runs pairwise TM-aligns on a folder of designs. Can subdivide designs by
# benchmark name or condition, to avoid doing unecessary combinatorial
# comparisons
# 

import sys, os, argparse, itertools, json, glob, re
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(script_dir+'util/')
import slurm_tools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=10000,help='How many tmalign comparisons to make in each job')
    parser.add_argument('--tmp_pre',type=str,default='tmalign.list', help='Name prefix of temporary files with lists of designs to score')
    parser.add_argument('--subdivide',type=str,default='prefix', choices=['none','task','cond','task+cond','prefix'],
        help='Only compute pairwise TM scores against designs with the same benchmark task and/or hyperparameter conditions.')
    parser.add_argument('-p', type=str, default='cpu',help='-p argument for slurm (partition)')
    parser.add_argument('-t', type=str, default='4:00:00',help='-t argument for slurm')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--wait_for',type=str, nargs='+', help='Space-separated slurm job IDs to wait for before starting the scoring jobs')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    args = parser.parse_args()

    # subdivide designs
    if args.subdivide == 'none':
        filenames_list = sorted(glob.glob(args.datadir+'/*.pdb'))
    elif args.subdivide == 'cond':
        pattern = re.compile('.*(cond\d+).*')
        conditions = np.unique([pattern.findall(fn)[0] for fn in glob.glob(args.datadir+'/*.pdb')])
        filenames_list = [(cond,sorted(glob.glob(args.datadir+f'*{cond}_*.pdb'))) for cond in conditions]
    elif args.subdivide == 'task':
        with open(script_dir+'benchmarks.json') as f: 
            benchmarks = json.load(f)
        filenames_list = [(bm,sorted(glob.glob(args.datadir+f'*{bm}*.pdb'))) for bm in benchmarks]
    elif args.subdivide == 'task+cond':
        with open(script_dir+'benchmarks.json') as f: 
            benchmarks = json.load(f)
        pattern = re.compile('.*(cond\d+).*')
        conditions = np.unique([pattern.findall(fn)[0] for fn in glob.glob(args.datadir+'/*.pdb')])
        filenames_list = [(f'{bm}_{cond}', sorted(glob.glob(args.datadir+f'/*{bm}_{cond}_*.pdb')))
            for bm in benchmarks for cond in conditions]
    elif args.subdivide == 'prefix':
        prefixes = set(['_'.join(os.path.basename(fn).split('_')[:-1]) 
            for fn in sorted(glob.glob(args.datadir+f'/*.pdb'))])
        filenames_list = [(f'{pre}', sorted(glob.glob(args.datadir+f'/{pre}_*.pdb')))
            for pre in prefixes]

    if sum([len(filenames) for filenames in filenames_list])==0: sys.exit('No pdbs to score. Exiting.')

    filenames_list = [x for x in filenames_list if len(x)>0]

    tmalign_folder = args.datadir+'/tmalign/'
    os.makedirs(tmalign_folder,exist_ok=True)

    # generate jobs
    job_fn = args.datadir + '/jobs.tmalign.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for label, filenames in filenames_list:
        pairs = [(f1,f2) for i,f1 in enumerate(filenames) for j,f2 in enumerate(filenames) if i<j]
        for i in np.arange(0, len(pairs), args.chunk):
            tmp_fn = tmalign_folder+f'/{args.tmp_pre}.{label}.{i}'
            with open(tmp_fn,'w') as outf:
                for pair in pairs[i:i+args.chunk]:
                    print(pair[0]+' '+pair[1], file=outf)
            print(f'python {script_dir}/util/tmalign_list.py {tmp_fn}', file=job_list_file)
    if args.submit: job_list_file.close()

    # submit job
    if args.submit: 
        if args.J is not None:
            job_name = args.J 
        else:
            job_name = 'tmalign_'+os.path.basename(args.datadir.strip('/'))
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres="", log=args.keep_logs, J=job_name, wait_for=args.wait_for, t=args.t)
        print(f'Submitted array job {slurm_job} for pairwise TM-align')

if __name__ == "__main__":
    main()
