#!/usr/bin/env python
#
# Generates and slurm array jobs for hyperparameter sweeps on design
# scripts, optionally submits array job and outputs slurm job ID
#

import sys, os, argparse, itertools, json, shutil
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(script_dir+'util/')
import slurm_tools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--command',type=str,help='design script to run')
    parser.add_argument('--args',type=str,nargs='+',required=True,help='string with all arguments to pass to the command, '\
                        'with colon-delimited value options for each')
    parser.add_argument('--benchmarks', type=str, nargs='+',help='Space-separated list of benchmark names, as defined in "benchmarks.json"')
    parser.add_argument('--num_per_condition', type=int, default=1,help='Number of designs to make for each condition')
    parser.add_argument('--num_per_job', type=int, default=1,help='Split runs for each condition into this many designs per job')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:rtx2080:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--out', type=str, default='out/out',help='Path prefix for output files')
    args = parser.parse_args()

    args_vals = [] # argument names and lists of values for passing to design script

    # default design script
    if args.command is None:
        args.command = os.path.abspath(script_dir+'../inpaint.py')

    # parse pre-defined benchmarks
    with open(script_dir+'benchmarks_inpaint.json') as f: 
        benchmarks = json.load(f)
    input_path = script_dir+'input/' # prepend path to input pdbs in current repo
    benchmark_list = []
    if args.benchmarks is not None:
        if args.benchmarks[0]=='all':
            to_run = benchmarks
        else:
            to_run = args.benchmarks
        for bm in to_run:
            pre = args.out if os.path.basename(args.out) == '' else args.out+'_'
            benchmark_list.append([f'--out {pre}{bm}', benchmarks[bm].replace('--pdb ','--pdb '+input_path)])

    # parse names of arguments and their value options to be passed into the design script
    arg_combos = []
    for argstr in args.args:
        args_vals = []

        for arg_val_str in (' '+argstr).split(' --')[1:]: # 1st element is empty string
            if ' ' in arg_val_str:
                i_space = arg_val_str.index(' ') 
                arg = arg_val_str[:i_space] # 1st space-delimited token is name of argument
                vals = arg_val_str[i_space+1:] # rest of string contains argument values delimited by :
                args_vals.append([f'--{arg} {val}' for val in vals.split('|')])
            else: # flag with no values
                args_vals.append([f'--{arg_val_str}'])

        arg_combos.extend([list(x) for x in itertools.product(*args_vals)])

    if len(benchmark_list) > 0:
        new_combos = []
        for benchmark in benchmark_list: # [output path, input pdb, contig spec]
            for i, arglist in enumerate(arg_combos):
                new_arglist = benchmark + arglist
                new_arglist[0] = new_arglist[0] + f'_cond{i}'
                new_combos.append(new_arglist)
        arg_combos = new_combos
    else:
        for i in range(len(arg_combos)):
            pre = args.out if os.path.basename(args.out) == '' else args.out+'_'
            arg_combos[i] = [f'--out {pre}cond{i}'] + arg_combos[i]

    # make output folder
    os.makedirs(os.path.dirname(args.out), exist_ok=True) 
    os.makedirs(os.path.dirname(args.out)+'/input', exist_ok=True) 

    # output commands with all combos of argument values
    job_fn = os.path.dirname(args.out) + '/jobs.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for icond, arglist in enumerate(arg_combos):
        extra_args = ' '.join(arglist)

        for istart in np.arange(0, args.num_per_condition, args.num_per_job):
            log_fn = arglist[0].replace('--out ','')+f'_{istart}.log'
            print(f'source activate SE3nv; python {args.command} {extra_args} '\
                  f'--num {args.num_per_job} --start_num {istart} >> {log_fn}', file=job_list_file)

        # copy input pdbs
        for argstr in arglist:
            if argstr.startswith('--pdb'):
                fn = argstr.split(' ')[1]
                outfn = os.path.dirname(args.out)+'/input/'+os.path.basename(fn)
                if not os.path.exists(outfn):
                    shutil.copyfile(fn, outfn)

    if args.submit: job_list_file.close()

    # submit job
    if args.submit:
        if args.J is not None:
            job_name = args.J 
        else:
            job_name = 'sweep_hyp_'+os.path.basename(os.path.dirname(args.out))
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name)
        print(slurm_job)

if __name__ == "__main__":
    main()
