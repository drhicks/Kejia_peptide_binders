#!/usr/bin/env python
#
# Compiles metrics from scoring runs into a single dataframe CSV
#

import os, argparse, glob
import numpy as np
import pandas as pd
from icecream import ic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs')
    parser.add_argument('--outcsv',type=str,default='compiled_metrics.csv',help='Output filename')
    args = parser.parse_args()

    filenames = glob.glob(args.datadir+'/*.trb')

    # load run metadata
    records = []
    for fn in filenames:
        name = os.path.basename(fn).replace('.trb','')
        trb = np.load(fn, allow_pickle=True)

        record = {'name':name}
        if 'lddt' in trb:
            record['lddt'] = trb['lddt'].mean()
        if 'inpaint_lddt' in trb:
            record['inpaint_lddt'] = np.mean(trb['inpaint_lddt'])

        if 'plddt' in trb:
            plddt = trb['plddt'].mean(1)
            record.update(dict(
                plddt_start = plddt[0],
                plddt_mid = plddt[len(plddt)//2],
                plddt_end = plddt[-1],
                plddt_mean = plddt.mean()
            ))
        if 'sampled_mask' in trb:
            record['sampled_mask'] = trb['sampled_mask']
        if 'config' in trb:
            for k,vdict in trb['config'].items():
                for k2,v in vdict.items():
                    record[k+'.'+k2] = v
        records.append(record)

    df = pd.DataFrame.from_records(records)

    # load computed metrics, if they exist
    for path in [
        args.datadir+'/af2_metrics.csv.*',
        args.datadir+'/pyrosetta_metrics.csv.*',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df = df.merge(tmp, on='name', how='outer')

    # mpnn metrics require adding an extra column
    df_mpnn = pd.DataFrame(dict(name=[]))
    for path in [
        args.datadir+'/mpnn/af2_metrics.csv.*',
        args.datadir+'/mpnn/pyrosetta_metrics.csv.*',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df_mpnn = df_mpnn.merge(tmp, on='name', how='outer')

    if os.path.exists(args.datadir+'/mpnn/'):
        mpnn_scores = load_mpnn_scores(args.datadir+'/mpnn/')
        df_mpnn = df_mpnn.merge(mpnn_scores, on='name', how='outer')
        df_mpnn['mpnn_index'] = df_mpnn.name.map(lambda x: int(x.split('_')[-1]))
        df_mpnn.name = df_mpnn.name.map(lambda x: '_'.join(x.split('_')[:-1]))
        df_mpnn = df_mpnn.merge(df, on='name', how='outer')
        print(df_mpnn['contig_rmsd_af2'])

    # combine regular and mpnn designs
    df_s = []
    if df.shape[1] > len(records[0].keys()):
        df['mpnn'] = False
        df_s.append(df)
    if df_mpnn.shape[1] > len(records[0].keys()):
        df_mpnn['mpnn'] = True
        df_s.append(df_mpnn)

    if len(df_s) == 0:
        df = pd.DataFrame.from_records(records)
    else:
        df = pd.concat(df_s)

    # add seq/struc clusters (assumed to be the same for mpnn designs as non-mpnn)
    for path in [
        args.datadir+'/tm_clusters.csv',
        args.datadir+'/blast_clusters.csv',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df = df.merge(tmp, on='name', how='outer')

    df.to_csv(args.datadir+'/'+args.outcsv)
    print(f'Wrote metrics dataframe {df.shape} to "{args.datadir}/{args.outcsv}"')

def load_mpnn_scores(folder):

    filenames = glob.glob(folder+'/seqs/*.fa')

    records = []
    for fn in filenames:
        scores = []
        with open(fn) as f:
            lines = f.readlines()
            for header in lines[2::2]:
                scores.append(float(header.split(',')[2].split('=')[1]))

            for i, score in enumerate(scores):
                records.append(dict(
                    name = os.path.basename(fn).replace('.fa','') + f'_{i}',
                    mpnn_score = score
                ))

    df = pd.DataFrame.from_records(records)
    return df

if __name__ == "__main__":
    main()
