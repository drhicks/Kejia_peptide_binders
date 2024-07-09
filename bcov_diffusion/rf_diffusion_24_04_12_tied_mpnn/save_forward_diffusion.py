import torch
import os, sys
import numpy as np
import argparse
import glob
import json
from parsers import parse_pdb
from diffusion import Diffuser
from util import writepdb, writepdb_multi
import progressbar

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(script_dir+'util/')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb",type=str,help='Path to pdb file to diffuse')
    parser.add_argument("--output_path",type=str,help="Path and prefix for files")
    parser.add_argument("--n_samples",type=int,help="n_samples",default=1)
    parser.add_argument("--diffusion_window",type=str, default=None, help='Window to keep fixed. E.g. 1:10,50:60')
    parser.add_argument("--T",type=int,default=200, help='T')
    parser.add_argument("--b_0",type=float,default=1e-2, help='b_0')
    parser.add_argument("--b_T",type=float,default=7e-2, help='b_T')
    parser.add_argument("--min_sigma",type=float,default=0.02, help='min_sigma')
    parser.add_argument("--max_sigma",type=float,default=1.5, help='max_sigma')
    parser.add_argument("--min_b",type=float,default=1.5, help='min_b')
    parser.add_argument("--max_b",type=float,default=2.5, help='max_b')
    parser.add_argument("--schedule_type",type=str,default="linear",help='schedule type')
    parser.add_argument("--so3_schedule_type",type=str,default="linear",help='so3 schedule type')
    parser.add_argument("--so3_type",type=str,default='igso3',help='so3 type')
    parser.add_argument("--chi_type",type=str, default='interp',help='chi type')
    parser.add_argument("--diff_crd_scale",type=float, default='0.0667',help='crd_scale')
    parser.add_argument("--aa_decode_steps",type=int, default=40, help='aa_decode_steps')
    args=parser.parse_args()

    diffuser = Diffuser(args.T, args.b_0, args.b_T, args.min_sigma, args.max_sigma, args.min_b, args.max_b, args.schedule_type, args.so3_schedule_type, so3_type=args.so3_type, chi_type=args.chi_type, crd_scale=args.diff_crd_scale, aa_decode_steps=args.aa_decode_steps)
    
    try:
        xyz,mask,idx,seq = parse_pdb(args.input_pdb, xyz27=True,seq=True)
    except:
        xyz,mask,idx,seq = parse_pdb(args.input_pdb, xyz27=False,seq=True)
    
    diffusion_mask = torch.zeros(xyz.shape[0]).bool()
    bfacts = torch.ones_like(torch.from_numpy(seq).squeeze())
    if args.diffusion_window is not None:
        parts = args.diffusion_window.split(",")
        for part in parts:
            diffusion_mask[int(part.split(":")[0]):int(part.split(":")[1])+1] = True
    
    # make output dir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    bar=progressbar.ProgressBar(maxval=args.n_samples, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(args.n_samples): 
        fa_stack,_,xyz_true = diffuser.diffuse_pose(torch.from_numpy(xyz),torch.from_numpy(seq),torch.from_numpy(mask),diffusion_mask=diffusion_mask, diffuse_sidechains=False)
        fa_stack = fa_stack.squeeze(1)
        fa_stack = torch.cat((xyz_true[None], fa_stack), dim=0)
        out = args.output_path+'_'+str(i)+'.pdb'
        writepdb_multi(out, fa_stack, bfacts, torch.from_numpy(seq).squeeze(), use_hydrogens=False, backbone_only=False)
        if i == 0:
            out_true = args.output_path+'_true'+'.pdb'
            writepdb(out_true,xyz_true, torch.from_numpy(seq))
        bar.update(i+1)
    bar.finish()
if __name__ == '__main__':
    main()

