#!/usr/bin/env python
#
# Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
# contig positions), makes list of MPNN jobs on batches of those designs,
# and optionally submits slurm array job and outputs job ID
# 

import sys, os, argparse, itertools, json, glob
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir,'util'))
import slurm_tools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=-1,help='How many designs to process in each job')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:a4000:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--cautious', action="store_true", default=False, help='Skip design if output file exists')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--num_seq_per_target', default=8,type=int, help='How many mpnn sequences per design? Default = 8')
    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')
    return args

def make_fixed_positions_dict(pdb_dict_list, folder):
    my_dict = {}
    for entry in pdb_dict_list:
        trb = np.load(glob.glob(folder+'/'+entry['name']+'.trb')[0],allow_pickle=True)
        my_dict[entry['name']] = {'A': [int(x[1]) for x in trb['con_hal_pdb_idx']]}
    return my_dict

def main():

    args = get_args()

    mpnn_folder = args.datadir+'/mpnn/'
    os.makedirs(mpnn_folder, exist_ok=True)

    filenames = glob.glob(args.datadir+'/*.pdb')
    
    # skip designs that have already been done
    if args.cautious:
        filenames = [fn for fn in filenames 
            if not os.path.exists(mpnn_folder+'/seqs/'+os.path.basename(fn).replace('.pdb','.fa'))]

    if args.chunk == -1:
        args.chunk = len(filenames)

    # run parser script
    job_fn = args.datadir + '/jobs.mpnn.parse.list'
    job_list_file = open(job_fn,'w')
    for i in range(0, len(filenames), args.chunk):
        with open(mpnn_folder+f'parse_multiple_chains.list.{i}','w') as outf:
            for fn in filenames[i:i+args.chunk]:
                print(fn,file=outf)
        print(f'python {script_dir}/util/parse_multiple_chains.py --input_files {mpnn_folder}/parse_multiple_chains.list.{i} '\
              f'--datadir {args.datadir} --output_parsed {mpnn_folder}/pdbs_{i}.jsonl '\
              f'--output_fixed_pos {mpnn_folder}/pdbs_position_fixed_{i}.jsonl', file=job_list_file)
    job_list_file.close()

    # submit to slurm
    if args.submit:
        slurm_job, proc = slurm_tools.array_submit(job_fn, p='cpu', gres=None, J='mpnn_pre', log=args.keep_logs)
        print(f'Submitted array job {slurm_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to preprocess {len(filenames)} designs for MPNN')

        prev_job = slurm_job
    else:
        prev_job = None

    job_fn = args.datadir + '/jobs.mpnn.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for i in range(0, len(filenames), args.chunk):
        print(f'source activate mlfold; python /home/jue/git/proteinmpnn/protein_mpnn_run.py '\
              f'--model_name "v_48_020" '\
              f'--jsonl_path {mpnn_folder}pdbs_{i}.jsonl '\
              f'--fixed_positions_jsonl {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
              f'--out_folder {mpnn_folder} '\
              f'--num_seq_per_target  {args.num_seq_per_target} '\
              f'--sampling_temp="0.1" '\
              f'--batch_size {8 if args.num_seq_per_target > 8 else args.num_seq_per_target} '\
              f'--omit_AAs XC',
              file=job_list_file)
    if args.submit: job_list_file.close()

    # submit job
    if args.submit:
        if args.J is not None:
            job_name = args.J
        else:
            job_name = 'mpnn_'+os.path.basename(args.datadir.strip('/'))
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name, wait_for=[prev_job])
        print(f'Submitted array job {slurm_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to MPNN {len(filenames)} designs')

if __name__ == "__main__":
    main()
