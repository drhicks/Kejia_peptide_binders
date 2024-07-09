#!/usr/bin/env python
#
# Parses TM-align results and saves TM score matrices and TM-score clusters
# 

import sys, os, argparse, glob
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs that TM align was run on')
    parser.add_argument('--thresh',type=int, nargs='+', default=[0.4,0.6,0.8],
        help='List of TM score thresholds to cluster on')
    args = parser.parse_args()

    tmalign_folder = args.datadir+'/tmalign/'
    filenames = glob.glob(tmalign_folder+'*.out')

    # parse separate TM matrices for different subdivisions of designs
    labels = np.unique([fn.split('.')[-3] for fn in filenames])

    df_s = []
    for label in labels:
        fn_s = [fn for fn in filenames if fn.split('.')[-3]==label]
        tm, names = load_tm_matrix(fn_s)
        df = pd.DataFrame()
        df['name'] = names

        clusters = []
        for thresh in args.thresh:
            clus = greedy_cluster(tm, thresh)
            clusters.append(clus)
            df[f'tm_cluster_{thresh:.2f}'] = [f'{label}_clus{c}' for c in clus]

        np.savez(tmalign_folder+f'tm_matrix.{label}.npz',
            tm=tm,
            names=names)

        df_s.append(df)

    df = pd.concat(df_s)

    df.to_csv(args.datadir+'/tm_clusters.csv')


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

def load_tm_matrix(fn_s):
    """Load a TM-score matrix from files of TMalign results.

    Input:
        fn_s: list of filenames which contain '<name1> <name2> <tm score>' on each line

    Output:
        tm: matrix of tm-scores
        names: list of structure names corresponding to the rows & columns
    """
    records = []
    for fn in fn_s:
        with open(fn) as f:
            lines = [l.strip().split() for l in f.readlines()]
        for l in lines:
            records.append([l[0],l[1],float(l[2])])

    names = np.unique([l[0] for l in records]+[l[1] for l in records])
    names = sorted(names)
    name2idx = dict(zip(names,range(len(names))))

    tm = np.full([len(names)]*2, 0.0)
    for k,row in enumerate(records):
        i = name2idx[row[0]]
        j = name2idx[row[1]]
        tm[i,j] = row[2]
        tm[j,i] = row[2]
    for i in range(len(tm)):
        tm[i,i] = 1

    return tm, names


if __name__ == "__main__":
    main()
