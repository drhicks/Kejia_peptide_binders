#!/usr/bin/env python
#
# Parses blast results and saves pairwise blast identity matrices and sequence clusters
# 

import sys, os, argparse, glob
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs that blast was run on')
    parser.add_argument('--thresh',type=int, nargs='+', default=[0.4,0.6,0.8],
        help='List of TM score thresholds to cluster on')
    args = parser.parse_args()

    tmalign_folder = args.datadir+'/blast/'
    filenames = glob.glob(tmalign_folder+'*.out')

    # parse separate TM matrices for different subdivisions of designs
    labels = np.unique([fn.split('.')[-3] for fn in filenames])

    df_s = []
    for label in labels:
        fn_s = [fn for fn in filenames if fn.split('.')[-3]==label]
        sim, names = load_sim_matrix(fn_s)
        df = pd.DataFrame()
        df['name'] = names

        clusters = []
        for thresh in args.thresh:
            clus = greedy_cluster(sim, thresh)
            clusters.append(clus)
            df[f'blast_cluster_{thresh:.2f}'] = [f'{label}_clus{c}' for c in clus]

        np.savez(tmalign_folder+f'blast_matrix.{label}.npz',
            blast_id=sim,
            names=names)

        df_s.append(df)

    df = pd.concat(df_s)

    df.to_csv(args.datadir+'blast_clusters.csv')


def greedy_cluster(sim, thresh):
    """Greedy clustering on similarity matrix `sim` with threshold `thresh`"""
    clus = np.full(sim.shape[0],-1)
    iclus = 0
    unassigned = np.where(clus==-1)[0]
    while len(unassigned) > 0:
        clus[(sim[unassigned[0],:] > thresh) & (clus==-1)] = iclus
        unassigned = np.where(clus==-1)[0]
        iclus += 1
    return clus

def load_sim_matrix(fn_s):
    """Load a blast sequence identity matrix from files of TMalign results.

    Input:
        fn_s: list of filenames which contain '<name1> <name2> <tm score>' on each line

    Output:
        sim: matrix of fraction identity scores
        names: list of structure names corresponding to the rows & columns
    """
    records = []
    for fn in fn_s:
        with open(fn) as f:
            lines = [l.strip().split(',') for l in f.readlines()]
        for l in lines:
            # qaccver saccver pident length mismatch gapopen qstart qend sstart send evalue bitscore nident qlen slen
            nident, qlen, slen = float(l[-3]), float(l[-2]), float(l[-1])
            records.append([l[0], l[1], nident/max(qlen,slen)])

    names = np.unique([l[0] for l in records]+[l[1] for l in records])
    names = sorted(names)
    name2idx = dict(zip(names,range(len(names))))

    sim = np.full([len(names)]*2, 0.0)
    for k,row in enumerate(records):
        i = name2idx[row[0]]
        j = name2idx[row[1]]
        sim[i,j] = row[2]
        sim[j,i] = row[2]
    for i in range(len(sim)):
        sim[i,i] = 1

    return sim, names


if __name__ == "__main__":
    main()
