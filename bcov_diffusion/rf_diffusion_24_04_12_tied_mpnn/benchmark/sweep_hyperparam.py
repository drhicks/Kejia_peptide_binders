#!/usr/bin/env python
#
# Generates and slurm array jobs for hyperparameter sweeps on design
# scripts, optionally submits array job and outputs slurm job ID
#

import sys, os, argparse, itertools, json, shutil
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(script_dir+'util/')
from icecream import ic 
import slurm_tools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--command',type=str,help='design script to run')
    parser.add_argument('--args',type=str,nargs='+',required=True,help='string with all arguments to pass to the command, '\
                        'with pipe (|)-delimited value options for each')
    parser.add_argument('--benchmarks', type=str, nargs='+',help='Space-separated list of benchmark names, as defined in "benchmarks.json"')
    parser.add_argument('--num_per_condition', type=int, default=1,help='Number of designs to make for each condition')
    parser.add_argument('--num_per_job', type=int, default=1,help='Split runs for each condition into this many designs per job')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-t', type=str, help='-t argument for slurm')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:rtx2080:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--out', type=str, default='out/out',help='Path prefix for output files')
    parser.add_argument('--benchmark_json', type=str, default='benchmarks.json', help='Path to non-standard custom json file of benchmarks')

    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')

    if args.num_per_job > args.num_per_condition:
        sys.exit('ERROR: --num_per_job cannot be greater than --num_per_condition '\
                 '(different conditions can\'t be in the same job.)')

    args_vals = [] # argument names and lists of values for passing to design script

    # default design script
    if args.command is None:
        args.command = os.path.abspath(script_dir+'../run_inference.py')

    # parse pre-defined benchmarks
    print('This is benchmarks json')
    print(args.benchmark_json)
    with open(script_dir+args.benchmark_json) as f: 
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
            benchmark_list.append([
                f'inference.output_prefix={pre}{bm}',
                benchmarks[bm].replace('inference.input_pdb=','inference.input_pdb='+input_path)
            ])

    # parse names of arguments and their value options to be passed into the design script
    arg_combos = []
    for argstr in args.args:
        args_vals = []

        i = 0
        tokens = argstr.split()
        while i<len(tokens):
            if '--config-name' in tokens[i] and '=' not in tokens[i]: # special case
                arg = tokens[i]
                vals = tokens[i+1]
                i += 2
            else:
                arg, vals = tokens[i].split('=')
                i += 1
            args_vals.append([f'{arg}={val}' for val in vals.split('|')])

        arg_combos.extend([list(x) for x in itertools.product(*args_vals)])

    # add benchmark-related arguments to the beginning of each argument list
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
            arg_combos[i] = [f'inference.output_prefix={pre}cond{i}'] + arg_combos[i]

    # make output folder
    os.makedirs(os.path.dirname(args.out), exist_ok=True) 
    os.makedirs(os.path.dirname(args.out)+'/input', exist_ok=True)

    # output commands with all combos of argument values
    job_fn = os.path.dirname(args.out) + '/jobs.list'
    ic(args.submit)
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for icond, arglist in enumerate(arg_combos):
        # log prefix is output prefix
        log_pre = arglist[0].replace('inference.output_prefix=','')

        # --config-name argument has to go first
        idx = np.where(['--config-name' in x for x in arglist])[0]
        if len(idx)>0:
            i = idx[0]
            arglist = [arglist[i]]+arglist[:i]+arglist[i+1:]

        extra_args = ' '.join(arglist)

        for istart in np.arange(0, args.num_per_condition, args.num_per_job):
            log_fn = log_pre+f'_{istart}.log'
            print(f'source activate SE3nv; python {args.command} {extra_args} '\
                  f'inference.num_designs={args.num_per_job} inference.design_startnum={istart} >> {log_fn}', file=job_list_file)

        # copy input pdbs
        for argstr in arglist:
            if argstr.startswith('inference.input_pdb'):
                fn = argstr.split(' ')[0].split('=')[1]
                outfn = os.path.dirname(args.out)+'/input/'+os.path.basename(fn)
                if not os.path.exists(outfn):
                    shutil.copyfile(fn, outfn)

    # submit job
    if args.submit:
        job_list_file.close()

        if args.J is not None:
            job_name = args.J
        else:
            job_name = 'sweep_hyp_'+os.path.basename(os.path.dirname(args.out))
        if args.p == 'cpu':
            args.gres = ""
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name, t=args.t)
        print(f'Submitted array job {slurm_job} with {len(arg_combos)*args.num_per_condition/args.num_per_job} jobs to make {len(arg_combos)*args.num_per_condition} designs')

        
if __name__ == "__main__":
    main()
