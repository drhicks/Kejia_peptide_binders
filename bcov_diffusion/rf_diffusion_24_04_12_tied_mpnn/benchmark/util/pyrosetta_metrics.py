#!/software/conda/envs/pyrosetta/bin/python

import pandas as pd
import numpy as np
import sys, os, argparse, glob
from collections import OrderedDict

import pyrosetta
pyrosetta.init('-mute all')

p = argparse.ArgumentParser()
p.add_argument('input_data', help='Folder of designs to process, or a file with a list of paths to designs')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
p.add_argument('-o','--outcsv', help='output csv file')
p.add_argument('--trb_dir', help='Folder containing .trb files (if not same as pdb folder)')
p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')
args = p.parse_args()

def get_topology(ss):
    topology = ''
    prev = None
    for x in ss:
        if (x != prev) and (x != 'L'):
            topology += x
        prev = x
    return topology

def main():

    # for radius of gyration
    rog_scorefxn = pyrosetta.ScoreFunction()
    rog_scorefxn.set_weight( pyrosetta.rosetta.core.scoring.ScoreType.rg , 1 )

    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()

    if args.template is not None:
        pose_ref_clean = pyrosetta.pose_from_file(args.template)
    else:
        last_template = ''

    # files to process
    if os.path.isdir(args.input_data):
        filenames = sorted(glob.glob(os.path.join(args.input_data,'*.pdb')))
    else:
        with open(args.input_data) as f:
            filenames = [l.strip() for l in f.readlines()]

    # input and output names
    if args.outcsv is None:
        args.outcsv = os.path.join(os.path.dirname(filenames[0]),'pyrosetta_metrics.csv')

    records = []
    for fn in filenames:
        row = OrderedDict()
        row['name'] = os.path.basename(fn).replace('.pdb','')

        pose_hal = pyrosetta.pose_from_file(fn)
        row['rog'] = rog_scorefxn( pose_hal )

        DSSP.apply(pose_hal)
        ss = pose_hal.secstruct()

        row['len'] = len(pose_hal.sequence())
        row['seq'] = pose_hal.sequence()
        row['net_charge'] = row['seq'].count('K')+row['seq'].count('R')\
                     -row['seq'].count('D')-row['seq'].count('E')
        row['ss'] = ss
        row['topology'] = get_topology(ss)
        row['ss_strand_frac'] = ss.count('E') / row['len']
        row['ss_helix_frac'] = ss.count('H') / row['len']
        row['ss_loop_frac'] = ss.count('L') / row['len']

        records.append(row)

    df = pd.DataFrame.from_records(records)

    df.to_csv(args.outcsv)

if __name__ == "__main__":
    main()
